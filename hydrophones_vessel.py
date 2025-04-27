import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.signal import chirp, correlate
from scipy.signal import butter, filtfilt
import asyncio
import websockets
import argparse
import json
import time


STEP_DURATION = 0.5  # seconds between updates

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

def generate_delayed_signals(fs, c, T, source_pos, hydrophones_pos, f0, f1, total_time):
    """
    Generate delayed sweep signals for an array of hydrophones based on the source position.
    
    Parameters:
        fs: Sampling rate (samples per second)
        c: Speed of sound in the medium (m/s)
        T: Duration of the sweep (seconds)
        source_pos: Position of the sound source (2D array or tuple [x, y])
        hydrophones_pos: List or array of positions of hydrophones (2D list of [x, y] positions)
        f0: Start frequency of the sweep (Hz)
        f1: End frequency of the sweep (Hz)
        total_time: Total duration of the signal (seconds)
    
    Returns:
        signals: List of numpy arrays, one per hydrophone
        time: Time vector (common for all signals)
    """
    num_samples = int(total_time * fs)
    time = np.arange(num_samples) / fs  # Time vector

    # Create the sweep signal
    sweep_samples = int(T * fs)
    sweep_t = np.arange(sweep_samples) / fs
    k = (f1 - f0) / T  # Sweep rate (Hz/s)
    sweep_signal = np.sin(2 * np.pi * (f0 * sweep_t + 0.5 * k * sweep_t**2))
    
    signals = []
    
    for hydrophone_pos in hydrophones_pos:
        # Calculate the distance between the hydrophone and the source
        dist = np.linalg.norm(np.array(source_pos) - np.array(hydrophone_pos))
        
        # Calculate the delay time in seconds and convert to samples
        delay_time = dist / c
        delay_samples = int(np.round(delay_time * fs))  # Delay in samples
        
        # Create an empty signal
        signal = np.zeros(num_samples)
        
        # Insert sweep after the delay
        if delay_samples + sweep_samples < num_samples:
            signal[delay_samples:delay_samples + sweep_samples] = sweep_signal
        else:
            # If sweep would go past the end, truncate it
            available_samples = num_samples - delay_samples
            if available_samples > 0:
                signal[delay_samples:] = sweep_signal[:available_samples]
        
        signals.append(signal)
    
    return np.array(signals), np.array(time)

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Bandpass filter that allows frequencies between lowcut and highcut to pass.
    
    Parameters:
        data: Input signal (time domain)
        fs: Sampling frequency
        lowcut: Low frequency of the bandpass filter
        highcut: High frequency of the bandpass filter
        order: Order of the filter
        
    Returns:
        filtered_data: The bandpass filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def add_bandlimited_noise_to_signals(signals, fs, f0, f1, noise_level=0.01):
    """
    Add band-limited Gaussian noise to the signals within the frequency range [f0, f1].
    
    Parameters:
        signals: List or array of generated signals (each hydrophone's signal)
        fs: Sampling frequency
        f0, f1: Frequency range for the band-limited noise
        noise_level: Standard deviation of the Gaussian noise (relative to signal amplitude)
        
    Returns:
        signals_with_noise: Signals with added band-limited Gaussian noise
    """
    # Get the shape of the signals (assuming signals is a 2D array where each row is a signal)
    num_hydrophones, num_samples = signals.shape
    
    # Generate random Gaussian noise
    noise = np.random.normal(0, noise_level, signals.shape)  # mean=0, std=noise_level, shape of the signals
    
    # Apply band-pass filter to the noise to limit it to the desired frequency range
    noise_with_bandpass = np.zeros_like(noise)
    for i in range(num_hydrophones):
        noise_with_bandpass[i] = bandpass_filter(noise[i], fs, f0, f1)
    
    # Add band-limited noise to the signals
    signals_with_noise = signals + noise_with_bandpass
    
    return signals_with_noise

lat = 63.4605
lng = 10.4051

fs = 10e6           # Sampling frequency
T = 200e-6          # Length of sweep
total_time = 4e-3   # Total length of signal
f0, f1 = 58e3, 72e3 # Start and stop frequency of the sweep
c = 1500            # Speed of sound in medium

# Positions of hydrophones
hydrophone1_pos = np.array([0.0, 0.2])
hydrophone2_pos = np.array([-1.0, 0.0])
hydrophone3_pos = np.array([1.0, 0.0])

# Grid
x_min, x_max = -3.5, 3.5
y_min, y_max = -3.5, 3.5
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
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1 , 1], figure=fig)

ax_main = fig.add_subplot(gs[:, 0])  # Full height for main plot
# Right column: use a separate GridSpec for stacked signal plots
gs_right = gridspec.GridSpecFromSubplotSpec(
    4, 1, 
    subplot_spec=gs[:, 1], 
    height_ratios=[1, 1, 1, 1],
    hspace=0.45   # << Add more vertical spacing here
)

ax_signal1 = fig.add_subplot(gs_right[0])
ax_signal2 = fig.add_subplot(gs_right[1])
ax_signal3 = fig.add_subplot(gs_right[2])
ax_corr    = fig.add_subplot(gs_right[3])

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

    source_pos = source.get_position()

    # Generate signals
    hydrophones_pos = [hydrophone1_pos, hydrophone2_pos, hydrophone3_pos]
    signals, time = generate_delayed_signals(fs, c, T, source_pos, hydrophones_pos, f0, f1, total_time)
    t = time

    signals = add_bandlimited_noise_to_signals(signals, fs, f0, f1, noise_level=0.5)

    # True delay (only used for simulation)
    d1 = np.linalg.norm(source_pos - hydrophone1_pos)
    time1 = d1 / c

    # Simulated signals
    hydrophone1 = signals[0]
    hydrophone2 = signals[1]
    hydrophone3 = signals[2]

    # Estimate TDOAs via cross-correlation
    corr12 = correlate(hydrophone2, hydrophone1, mode='full')
    lags12 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
    peak12 = np.argmax(np.abs(corr12))
    estimated_dt12 = lags12[peak12] / fs

    corr13 = correlate(hydrophone3, hydrophone1, mode='full')
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
    tolerance = 0.005
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
        hyperbola12_plot.remove()
    if hyperbola13_plot:
        hyperbola13_plot.remove()
    if circle_plot:
        circle_plot.remove()

    hyperbola12_plot = ax_main.contour(xx, yy, diff12 - level12, levels=[0], linestyles='--', colors='blue')
    hyperbola13_plot = ax_main.contour(xx, yy, diff13 - level13, levels=[0], linestyles='--', colors='purple')
    circle_plot = ax_main.contour(xx, yy, circle_eq, levels=[0], linestyles='--', colors='green')

    # --- Update signal plots individually ---
    ax_signal1.clear()
    ax_signal1.plot(t * 1e3, hydrophone1, color='blue')
    ax_signal1.set_title("Hydrofon 1")
    ax_signal1.set_ylabel("Amplitude")
    ax_signal1.grid(True)

    ax_signal2.clear()
    ax_signal2.plot(t * 1e3, hydrophone2, color='orange')
    ax_signal2.set_title("Hydrofon 2")
    ax_signal2.set_ylabel("Amplitude")
    ax_signal2.grid(True)

    ax_signal3.clear()
    ax_signal3.plot(t * 1e3, hydrophone3, color='green')
    ax_signal3.set_title("Hydrofon 3")
    ax_signal3.set_ylabel("Amplitude")
    ax_signal3.set_xlabel("Tid (ms)")
    ax_signal3.grid(True)

    # --- Update right-side correlation plot ---
    ax_corr.clear()
    ax_corr.plot(lags12 / fs * 1e6, corr12, label='H1-H2', color='blue')
    ax_corr.axvline(x=lags12[peak12] / fs * 1e6, color='blue', linestyle='--')

    ax_corr.plot(lags13 / fs * 1e6, corr13, label='H1-H3', color='green')
    ax_corr.axvline(x=lags13[peak13] / fs * 1e6, color='green', linestyle='--')

    ax_corr.set_title(r"Krysskorrelasjon")
    ax_corr.set_xlabel("Tidsforsinkelse (µs)")
    ax_corr.set_ylabel("Korrelasjon")
    ax_corr.grid(True)
    ax_corr.legend()


    fig.canvas.draw_idle()



async def run_plots():
    # Start animation
    ani = FuncAnimation(fig, update, interval=STEP_DURATION*1000)

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