import websockets
import argparse
import asyncio
import json


vessel_connections: set[websockets.WebSocketServerProtocol] = set()
viewer_connections: set[websockets.WebSocketServerProtocol] = set()

vessel_data: dict[dict] = {}


async def broadcast_updates(update_queue: asyncio.Queue):
    """Wait for updates and broadcast them to all viewers"""
    while True:
        id, data = await update_queue.get()

        # Update global state
        vessel_data[id] = data

        message = json.dumps({
            "type": "vessel_update",
            "data": data
        })

        # Send to all connected viewers
        dead = set()
        for viewer in viewer_connections:
            try:
                await viewer.send(message)
            except:
                dead.add(viewer)

        # Clean up disconnected viewers
        for v in dead:
            viewer_connections.discard(v)


async def handler(websocket: websockets.WebSocketServerProtocol, update_queue: asyncio.Queue):
    """Handler for new connections"""
    try:
        init_message = await websocket.recv()
        init_data: dict = json.loads(init_message)

        client_type = init_data.get("type")

        if client_type == "vessel":
            id = init_data.get("id")
            if not id:
                await websocket.send(json.dumps({"error": "vessel must have an ID"}))
                return

            vessel_connections.add(websocket)
            vessel_data[id] = init_data
            print(f"vessel {id} connected: {websocket.remote_address}")

            async for message in websocket:
                try:
                    data: dict = json.loads(message)
                    # if data.get("timestamp") > boat_data.get("timestamp"):
                    await update_queue.put((id, data))
                except json.JSONDecodeError:
                    print(f"Invalid JSON data received from vessel {id}: {message}")


        elif client_type == "viewer":
            viewer_connections.add(websocket)
            print(f"Viewer connected: {websocket.remote_address}")

            # Send full boat data once
            full_update = json.dumps({
                "type": "full_update",
                "vessels": vessel_data
            })
            await websocket.send(full_update)

            try:
                await websocket.wait_closed()
            finally:
                print(f"Viewver disconnected: {websocket.remote_address}")
                viewer_connections.discard(websocket)

        else:
            print(f"Invalid client type tried to connect: {websocket.remote_address}")
            await websocket.send(json.dumps({"error": "Invalid client type"}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except json.JSONDecodeError:
        print("Invalid JSON from client", init_message)
    finally:
        if websocket in viewer_connections:
            viewer_connections.discard(websocket)
        if websocket in vessel_connections:
            vessel_connections.discard(websocket)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    update_queue = asyncio.Queue()

    async def handler_with_queue(websocket):
        await handler(websocket, update_queue)

    async with websockets.serve(handler_with_queue, args.host, args.port):
        await asyncio.gather(
            broadcast_updates(update_queue),
            asyncio.Future(),  # Keeps the server alive
        )


asyncio.run(main())
