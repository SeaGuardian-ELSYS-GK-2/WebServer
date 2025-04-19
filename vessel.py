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


async def vessel_client(vessel_id: str, path_file: str, host: str, port: int):
    uri = f"ws://{host}:{port}"

    # Load config file (coordinates and crew)
    with open(path_file, 'r') as f:
        config = json.load(f)

    if "coordinates" not in config or not isinstance(config["coordinates"], list):
        raise ValueError("Invalid config file: missing or invalid 'coordinates' list")

    if "crew" not in config or not isinstance(config["crew"], dict):
        raise ValueError("Invalid config file: missing or invalid 'crew' dict")

    coordinates = config["coordinates"]
    crew = config["crew"]

    if len(coordinates) < 2:
        raise ValueError("Config error: 'coordinates' must contain at least 2 points.")

    async with websockets.connect(uri) as websocket:
        # Initial message
        await websocket.send(json.dumps({
            "type": "vessel",
            "id": vessel_id,
            "timestamp": time.time(),
            "crew": crew
        }))

        current_index = 0

        while True:
            start = coordinates[current_index]
            end = coordinates[(current_index + 1) % len(coordinates)]

            for step in range(STEPS_PER_SEGMENT):
                alpha = step / STEPS_PER_SEGMENT
                lat = (1 - alpha) * start["lat"] + alpha * end["lat"]
                lng = (1 - alpha) * start["lng"] + alpha * end["lng"]

                update = {
                    "id": vessel_id,
                    "timestamp": time.time(),
                    "lat": lat,
                    "lng": lng,
                    "crewupdates": {}  # could be used for live updates of crew if needed later
                }

                await websocket.send(json.dumps(update))
                await asyncio.sleep(STEP_DURATION)

            current_index = (current_index + 1) % len(coordinates)


asyncio.run(vessel_client(args.id, args.path, args.host, args.port))