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


async def handle_init_message(websocket: websockets.WebSocketServerProtocol):
    """Handles the init message and parses it as json. Also closes the connection if an error occurs"""
    try:
        init_message = await asyncio.wait_for(websocket.recv(), timeout=5)
        init_data: dict = json.loads(init_message)
        return init_data
    except asyncio.TimeoutError:
        print(f"Client at {websocket.remote_address} failed to send init message in time.")
        await websocket.send(json.dumps({"error": "Timeout waiting for init message"}))
        await websocket.close()
        return None
    except json.JSONDecodeError:
        print(f"Client at {websocket.remote_address} sent invalid JSON: {init_message}")
        await websocket.send(json.dumps({"error": "Invalid JSON"}))
        await websocket.close()
        return None


async def handle_vessel_connection(websocket: websockets.WebSocketServerProtocol, update_queue: asyncio.Queue, init_data: dict):
    """Handles a connection to a vessel"""
    id = init_data.get("id")
    if not id:
        await websocket.send(json.dumps({"error": "vessel must have an ID"}))
        return

    vessel_connections.add(websocket)
    vessel_data[id] = init_data
    print(f"Client at {websocket.remote_address} identified as Vessel {id}")

    async for message in websocket:
        try:
            data: dict = json.loads(message)
            # if data.get("timestamp") > boat_data.get("timestamp"):
            await update_queue.put((id, data))
        except json.JSONDecodeError:
            print(f"Invalid JSON data received from vessel {id}: {message}")


async def handle_viewer_connection(websocket: websockets.WebSocketServerProtocol):
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


async def handler(websocket: websockets.WebSocketServerProtocol, update_queue: asyncio.Queue):
    """Handler for new connections"""
    try:
        print(f"Client connected: {websocket.remote_address}")

        init_data = await handle_init_message(websocket)
        if not init_data:
            return

        client_type = init_data.get("type")

        if client_type == "vessel":
            await handle_vessel_connection(websocket, update_queue, init_data)

        elif client_type == "viewer":
            await handle_viewer_connection(websocket, update_queue)

        else:
            print(f"Invalid client type tried to connect: {websocket.remote_address}")
            await websocket.send(json.dumps({"error": f"Invalid client type: {client_type}"}))
            websocket.close()

    except websockets.exceptions.ConnectionClosed:
        pass
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
