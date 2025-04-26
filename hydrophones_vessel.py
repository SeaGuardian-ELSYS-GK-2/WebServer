import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.signal import chirp, correlate
from scipy.ndimage import shift
import asyncio
import websockets
import argparse
import json
import time


STEP_DURATION = 2  # seconds between updates

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="5")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)

args = parser.parse_args()
uri = f"ws://{args.host}:{args.port}"

vessel_lat = 63.4405
vessel_lng = 10.3851

crew_member_lat = vessel_lat
crew_member_lng = vessel_lng

new_estimated_x = None
new_estimated_y = None

crew_json = '''
{
    "5_1": {
        "name": "Katrine",
        "overBoard": false
    }
}
'''

crew = json.loads(crew_json)

fs = 100e3
T = 10e-3
f0, f1 = 38e3, 42e3
c = 1500

hydrophone1_pos = np.array([0.0, 0.5])
hydrophone2_pos = np.array([-1.0, 0.0])
hydrophone3_pos = np.array([1.0, 0.0])

x_min, x_max = -10, 10
y_min, y_max = -10, 10
Ngrid = 600
x_vals = np.linspace(x_min, x_max, Ngrid)
y_vals = np.linspace(y_min, y_max, Ngrid)
xx, yy = np.meshgrid(x_vals, y_vals)

t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

def time_to_samples(tdelay):
    return int(round(tdelay * fs))


# --- Draggable point class ---
class DraggablePoint:
    def __init__(self, ax, pos, **plot_kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.pos = pos  # expects a NumPy array [x, y]
        self.point, = ax.plot([self.pos[0]], [self.pos[1]], 'o', picker=5, **plot_kwargs)
        self.dragging = False

        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.point.contains(event)
        if contains:
            self.dragging = True

    def on_release(self, event): # Release does not work without event
        self.dragging = False

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.pos[0], self.pos[1] = event.xdata, event.ydata
        self.point.set_data([self.pos[0]], [self.pos[1]])
        self.canvas.draw_idle()

    def get_position(self):
        return self.pos.copy()


def offset_latlng(lat, lng, dx_m, dy_m):
    R = 6378137 # Earth radius in meters

    lat_rad = np.radians(lat)

    dlat = dy_m / R
    dlng = dx_m / (R * np.cos(lat_rad))

    new_lat = lat + np.degrees(dlat)
    new_lng = lng + np.degrees(dlng)

    return new_lat, new_lng


# --- Set up figure with subplots ---
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)

ax_main = fig.add_subplot(gs[:, 0])  # Full height for main plot
ax_signal = fig.add_subplot(gs[0, 1])  # Top right: Signals
ax_corr = fig.add_subplot(gs[1, 1])    # Bottom right: Correlation

ax_main.set_xlim(x_min, x_max)
ax_main.set_ylim(y_min, y_max)
ax_main.set_aspect('equal')
ax_main.grid(True)
ax_main.set_title("TDOA-basert lokalisering med interseksjon")

# Hydrophones
ax_main.plot(*hydrophone1_pos, 'o', label=f"H1 ({hydrophone1_pos[0]:.2f},{hydrophone1_pos[1]:.2f})")
ax_main.plot(*hydrophone2_pos, 'o', label=f"H2 ({hydrophone2_pos[0]:.2f},{hydrophone2_pos[1]:.2f})")
ax_main.plot(*hydrophone3_pos, 'o', label=f"H3 ({hydrophone3_pos[0]:.2f},{hydrophone3_pos[1]:.2f})")

# Source
source = DraggablePoint(ax_main, np.array([1.75, 0.75]), color='red')

# Intersection and contours
intersection_plot, = ax_main.plot([], [], 'b*', label="Interseksjon", markersize=8)
hyperbola12_plot = None
hyperbola13_plot = None
circle_plot = None

ax_main.legend()


def update(_):
    global hyperbola12_plot, hyperbola13_plot, circle_plot
    global crew_member_lat, crew_member_lng
    # global new_estimated_x, new_estimated_y

    source_pos = source.get_position()

    # True delay (only used for simulation)
    d1 = np.linalg.norm(source_pos - hydrophone1_pos)
    d2 = np.linalg.norm(source_pos - hydrophone2_pos)
    d3 = np.linalg.norm(source_pos - hydrophone3_pos)
    time1 = d1 / c

    delta_t12_true = (d2 - d1) / c
    delta_t13_true = (d3 - d1) / c

    # Simulated signals
    hydrophone1 = sweep
    hydrophone2 = shift(sweep, -time_to_samples(delta_t12_true), mode='nearest')
    hydrophone3 = shift(sweep, -time_to_samples(delta_t13_true), mode='nearest')

    # Estimate TDOAs via cross-correlation
    corr12 = correlate(hydrophone1, hydrophone2, mode='full')
    lags12 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
    peak12 = np.argmax(np.abs(corr12))
    estimated_dt12 = lags12[peak12] / fs

    corr13 = correlate(hydrophone1, hydrophone3, mode='full')
    lags13 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
    peak13 = np.argmax(np.abs(corr13))
    estimated_dt13 = lags13[peak13] / fs

    print(f"[Update] Estimert TDOA H2-H1: {estimated_dt12 * 1e6:.2f} µs")
    print(f"[Update] Estimert TDOA H3-H1: {estimated_dt13 * 1e6:.2f} µs")

    # Geometry for plotting
    d1_grid = np.sqrt((xx - hydrophone1_pos[0])**2 + (yy - hydrophone1_pos[1])**2)
    d2_grid = np.sqrt((xx - hydrophone2_pos[0])**2 + (yy - hydrophone2_pos[1])**2)
    d3_grid = np.sqrt((xx - hydrophone3_pos[0])**2 + (yy - hydrophone3_pos[1])**2)

    diff12 = d2_grid - d1_grid
    diff13 = d3_grid - d1_grid

    level12 = c * estimated_dt12
    level13 = c * estimated_dt13

    circle_radius = c * time1
    circle_eq = d1_grid - circle_radius

    # Intersections
    tolerance = 0.01
    intersection_mask = (np.abs(diff12 - level12) < tolerance) & (np.abs(diff13 - level13) < tolerance) & (np.abs(circle_eq) < tolerance)
    intersection_x = xx[intersection_mask]
    intersection_y = yy[intersection_mask]
    intersection_plot.set_data(intersection_x, intersection_y)

    if intersection_x.size > 0:
        estimated_x = np.mean(intersection_x)
        estimated_y = np.mean(intersection_y)

        crew_member_lat, crew_member_lng = offset_latlng(vessel_lat, vessel_lng, estimated_x, estimated_y)

        print(f"[Update] Beregnet koordinater: ({crew_member_lat:.6f}, {crew_member_lng:.6f})")
    else:
        print("[Update] Ingen interseksjon funnet.")

    # Plot update for main figure
    if hyperbola12_plot:
        [c.remove() for c in hyperbola12_plot.collections]
    if hyperbola13_plot:
        [c.remove() for c in hyperbola13_plot.collections]
    if circle_plot:
        [c.remove() for c in circle_plot.collections]

    hyperbola12_plot = ax_main.contour(xx, yy, diff12 - level12, levels=[0], linestyles='--', colors='blue')
    hyperbola13_plot = ax_main.contour(xx, yy, diff13 - level13, levels=[0], linestyles='--', colors='purple')
    circle_plot = ax_main.contour(xx, yy, circle_eq, levels=[0], linestyles='--', colors='green')

    # --- Update right-side signal plot ---
    ax_signal.clear()
    ax_signal.plot(t * 1e3, hydrophone1, label="Hydrophone 1")
    ax_signal.plot(t * 1e3, hydrophone2, label="Hydrophone 2")
    ax_signal.set_title("Simulerte signaler")
    ax_signal.set_xlabel("Tid (ms)")
    ax_signal.legend()
    ax_signal.grid(True)

    # --- Update right-side correlation plot ---
    ax_corr.clear()
    ax_corr.plot(lags12 / fs * 1e6, corr12)
    ax_corr.axvline(x=lags12[peak12] / fs * 1e6, color='red', linestyle='--', label=f'Maks korrelasjon ({estimated_dt12*1e6:.2f} µs)')
    ax_corr.set_title(r"Krysskorrelasjon")
    ax_corr.set_xlabel("Tidsforsinkelse (µs)")
    ax_corr.grid(True)
    ax_corr.legend()

    fig.canvas.draw_idle()



async def run_plots():
    # Start animation
    ani = FuncAnimation(fig, update, interval=2000)

    # Instead of plt.show(), run event loop manually
    while True:
        plt.pause(0.01)
        await asyncio.sleep(0.01)


# --- WebSocket vessel sender ---
async def vessel_client(vessel_id: str, host: str, port: int):
    global crew_member_lat, crew_member_lng

    uri = f"ws://{host}:{port}"

    async with websockets.connect(uri) as websocket:
        # Initial crew info
        await websocket.send(json.dumps({
            "type": "vessel",
            "id": vessel_id,
            "timestamp": time.time(),
            "crew": crew
        }))

        while True:
            # If new estimation available, update position
            # if new_estimated_x is not None and new_estimated_y is not None:
            #     crew_member_lat, crew_member_lng = offset_latlng(vessel_lat, vessel_lng, new_estimated_x, new_estimated_y)

            update_data = {
                "id": vessel_id,
                "timestamp": time.time(),
                "lat": vessel_lat,
                "lng": vessel_lng,
                "crewupdates": {
                    "5_1": {
                        "overBoard": True,
                        "latitude": crew_member_lat,
                        "longitude": crew_member_lng
                    }
                }
            }

            await websocket.send(json.dumps(update_data))
            await asyncio.sleep(STEP_DURATION)


# --- Run both tasks ---
async def main():
    await asyncio.gather(
        run_plots(),
        vessel_client(args.id, args.host, args.port)
    )


asyncio.run(main())