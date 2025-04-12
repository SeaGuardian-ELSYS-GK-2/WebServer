import asyncio
import websockets
import json
import time


async def boat_client():
    uri = "ws://10.0.0.7:8000"

    async with websockets.connect(uri) as websocket:
        # Identify as a boat with ID "boat_1"
        await websocket.send(json.dumps({
            "type": "vessel",
            "id": 1,
            "timestamp": time.time()
        }))

        while True:
            data = {
                "x": 42,
                "y": 17,
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(data))
            await asyncio.sleep(2)


asyncio.run(boat_client())
