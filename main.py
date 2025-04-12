import subprocess
import time
import os
import sys
import socket


def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP address: {e}")
        sys.exit()


def launch_server(ip, port, log_file):
    return subprocess.Popen(
        ["python", "-u", "server.py", "--host", ip, "--port", str(port)],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT
    )


def launch_vessel(vessel_id, path, ip, port, log_dir):
    return subprocess.Popen(
        ["python", "-u", "vessel.py",
         "--id", str(vessel_id),
         "--path", path,
         "--host", ip,
         "--port", str(port)],
        stdout=open(os.path.join(log_dir, f"vessel_{vessel_id}.log"), "w"),
        stderr=subprocess.STDOUT
    )


def main():
    os.makedirs("logs", exist_ok=True)

    ip = get_ip_address()
    port = 8000

    print(f"Starting server at ws://{ip}:{port}")
    server_process = launch_server(ip, port, "logs/server.log")

    time.sleep(1)  # Give the server time to start
    if server_process.poll() is not None:
        print("Server process exited early! Check logs.")

    vessel_configs = [
        (1, "vessel_paths/1.json"),
        (2, "vessel_paths/2.json"),
        (3, "vessel_paths/3.json"),
    ]

    vessels = []
    for vessel_id, path in vessel_configs:
        print(f"Starting vessel {vessel_id}")
        proc = launch_vessel(vessel_id, path, ip, port, "logs")
        time.sleep(0.5)  # Give it a moment to initialize
        if proc.poll() is not None:
            print(f"Vessel {vessel_id} exited early! Check logs/vessel_{vessel_id}.log.")
        else:
            vessels.append(proc)

    print("All processes launched. Press Ctrl+C to stop.")
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_process.terminate()
        for p in vessels:
            p.terminate()


main()
