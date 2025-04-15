import asyncio
import websockets
import json
import time
import argparse


STEP_DURATION = 2       # seconds between updates
SEGMENT_DURATION = 60   # seconds to travel from one point to the next
STEPS_PER_SEGMENT = SEGMENT_DURATION // STEP_DURATION


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)

args = parser.parse_args()
uri = f"ws://{args.host}:{args.port}"


async def vessel_client(vessel_id: str, path_file: str, host: str, port: str):
    uri = f"ws://{host}:{port}"

    # Load path
    with open(path_file, 'r') as f:
        path = json.load(f)

    if len(path) < 2:
        print("Path must contain at least 2 points.")
        return

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "type": "vessel",
            "id": vessel_id,
            "timestamp": time.time()
        }))

        current_index = 0

        while True:
            # Loop back to start when reaching the end
            start = path[current_index]
            end = path[(current_index + 1) % len(path)]

            for step in range(STEPS_PER_SEGMENT):
                alpha = step / STEPS_PER_SEGMENT
                lat = (1 - alpha) * start["lat"] + alpha * end["lat"]
                lng = (1 - alpha) * start["lng"] + alpha * end["lng"]

                data = {
                    "id": vessel_id,
                    "timestamp": time.time(),
                    "lat": lat,
                    "lng": lng
                }

                await websocket.send(json.dumps(data))
                await asyncio.sleep(STEP_DURATION)

            current_index = (current_index + 1) % len(path)


asyncio.run(vessel_client(args.id, args.path, args.host, args.port))
