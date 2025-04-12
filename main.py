import asyncio
import json
import websockets
import socket


boats: set[websockets.WebSocketServerProtocol] = set()
viewers: set[websockets.WebSocketServerProtocol] = set()

boat_data = {}


def get_ip_address():
    """Returns the IP address of the server"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "Unknown"


async def broadcast_updates(update_queue: asyncio.Queue):
    """Wait for updates and broadcast them to all viewers."""
    while True:
        boat_id, data = await update_queue.get()

        # Update global state
        boat_data[boat_id] = data

        message = json.dumps({
            "type": "boat_update",
            "id": boat_id,
            "position": data
        })

        # Send to all connected viewers
        dead = set()
        for viewer in viewers:
            try:
                await viewer.send(message)
            except:
                dead.add(viewer)

        # Clean up disconnected viewers
        for v in dead:
            viewers.discard(v)


async def handler(websocket: websockets.WebSocketServerProtocol, update_queue: asyncio.Queue):
    """Handler for new connections"""
    try:
        init_message = await websocket.recv()
        data: dict = json.loads(init_message)

        client_type = data.get("type")

        if client_type == "boat":
            boat_id = data.get("id")
            if not boat_id:
                await websocket.send(json.dumps({"error": "Boat must have an ID"}))
                return

            boats.add(websocket)
            print(f"Boat {boat_id} connected: {websocket.remote_address}")

            async for message in websocket:
                try:
                    data: dict = json.loads(message)
                    if data.get("timestamp") > boat_data.get("timestamp"):
                        await update_queue.put((boat_id, data))
                except json.JSONDecodeError:
                    print(f"Invalid JSON data received from boat {boat_id}: {message}")


        elif client_type == "viewer":
            viewers.add(websocket)
            print(f"Viewer connected: {websocket.remote_address}")

            # Send full boat data once
            full_update = json.dumps({
                "type": "full_update",
                "boats": boat_data
            })
            await websocket.send(full_update)

            try:
                await websocket.wait_closed()
            finally:
                viewers.discard(websocket)

        else:
            print(f"Invalid client type tried to connect: {websocket.remote_address}")
            await websocket.send(json.dumps({"error": "Invalid client type"}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except json.JSONDecodeError:
        print("Invalid JSON from client", init_message)
    finally:
        if websocket in viewers:
            viewers.discard(websocket)
        if websocket in boats:
            boats.discard(websocket)


async def main():
    local_ip = get_ip_address()
    print(f"WebSocket server running on ws://{local_ip}:8000")

    update_queue = asyncio.Queue()

    async def handler_with_queue(websocket):
        await handler(websocket, update_queue)

    async with websockets.serve(handler_with_queue, "0.0.0.0", 8000):
        await asyncio.gather(
            broadcast_updates(update_queue),
            asyncio.Future(),  # Keeps the server alive
        )


asyncio.run(main())
