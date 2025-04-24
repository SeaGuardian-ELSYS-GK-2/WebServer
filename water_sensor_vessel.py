import serial
import serial.tools.list_ports
import asyncio
import websockets
import argparse
import json
import time
import re

STEP_DURATION = 0.05  # seconds between updates
OVERBOARD_THRESHOLD = 900  # mV

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="4")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)

args = parser.parse_args()
uri = f"ws://{args.host}:{args.port}"

# Hardcoded values
vessel_lat = 63.4305
vessel_lng = 10.3951

crew_member_lat = 63.4405
crew_member_lng = 10.3901

crew_json = '''
{
    "4_1": {
        "name": "John",
        "overBoard": false
    }
}
'''

crew = json.loads(crew_json)

print(crew)

def find_usb_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usb" in port.device.lower() or "tty.usb" in port.device.lower():
            print(f"Found USB device: {port.device}")
            return port.device
    raise Exception("No USB serial device found.")

async def read_serial_voltage():
    global latest_voltage, overboard_status

    try:
        port_name = find_usb_serial_port()
        ser = serial.Serial(port=port_name, baudrate=9600, timeout=1)
        print(f"Connected to {port_name}")

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                match = re.search(r"Voltage:\s*(\d+)\s*mV", line)
                if match:
                    latest_voltage = int(match.group(1))
                    overboard_status = latest_voltage < OVERBOARD_THRESHOLD
                    print(f"[Voltage] {latest_voltage} mV → Overboard: {overboard_status}")
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"[Serial Error] {e}")

# --- WebSocket vessel sender ---
async def vessel_client(vessel_id: str, host: str, port: int):
    global new_status
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
            if overboard_status:
                update = {
                    "id": vessel_id,
                    "timestamp": time.time(),
                    "lat": vessel_lat,
                    "lng": vessel_lng,
                    "crewupdates": {
                        "4_1": {
                            "overBoard": overboard_status,
                            "latitude": crew_member_lat,
                            "longitude": crew_member_lng
                        }
                    }
                }
            else:
                update = {
                    "id": vessel_id,
                    "timestamp": time.time(),
                    "lat": vessel_lat,
                    "lng": vessel_lng,
                    "crewupdates": {
                        "4_1": {
                            "overBoard": overboard_status
                        }
                    }
                }

            await websocket.send(json.dumps(update))
            # print(f"[WebSocket] Sent: {update}")
            await asyncio.sleep(STEP_DURATION)

# --- Run both tasks ---
async def main():
    await asyncio.gather(
        read_serial_voltage(),
        vessel_client(args.id, args.host, args.port)
    )

asyncio.run(main())