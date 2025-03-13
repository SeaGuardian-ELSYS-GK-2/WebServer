import asyncio
import json
import serial
import websockets
import socket

# ser = serial.Serial('/dev/cu.usbserial-210', 115200, timeout=1)

boat_x, boat_y = 63.417878, 10.402178

clients = set()


x_offset = 0
y_offset = 0


def get_ip_address():
    """Get the local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "Unknown"


async def send_position():
    """Reads data from serial and broadcasts it via WebSocket."""
    global boat_x, boat_y, x_offset, y_offset
    while True:
        try:
            # line = ser.readline().decode('utf-8').strip()
            # if line:
                # parts = line.split()

                # if len(parts) != 2:
                #     raise ValueError

                # x_offset, y_offset = map(float, parts)

                x_offset += 1
                y_offset += 1


                sea_guardian_x = boat_x + x_offset/1000
                sea_guardian_y = boat_y + y_offset/1000

                position = {"lat": sea_guardian_x, "lng": sea_guardian_y}
                print(f"Broadcasting: {position}")

                # Broadcast position to all connected clients
                if clients:
                    await asyncio.gather(*(client.send(json.dumps(position)) for client in clients))

        except ValueError as ve:
            print(f"Error parsing serial data: {ve}")
        except Exception as e:
            print(f"Error reading serial: {e}")

        await asyncio.sleep(0.5)


async def handler(websocket):
    """Handles WebSocket connections."""
    clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received: {message}")
    except websockets.exceptions.ConnectionClosedError:
        print("Client disconnected abruptly")
    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected")
    finally:
        clients.remove(websocket)


async def main():
    local_ip = get_ip_address()
    print(f"WebSocket server running on ws://{local_ip}:8000")

    async with websockets.serve(handler, "0.0.0.0", 8000) as server:
        await asyncio.gather(send_position(), server.wait_closed())


# Run main
asyncio.run(main())