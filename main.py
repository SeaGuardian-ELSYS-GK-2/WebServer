import subprocess
import time
import os
import sys
import socket
from rich.console import Console

HOST = "0.0.0.0"
PORT = 8000


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


def launch_server(ip, port):
    return subprocess.Popen(
        ["python", "-u", "server.py", "--host", ip, "--port", str(port)],
        stdout=None,
        stderr=None
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
    console = Console()

    print(f"Starting server: {HOST}:{PORT}")
    server_process = launch_server(HOST, PORT)

    ip = get_ip_address()
    console.print(f"Server can be reached from [bold blue]ws://{ip}:{PORT}[/]")

    time.sleep(0.5)  # Give the server time to start
    if server_process.poll() is not None:
        console.print("Server process exited early!", style="bold red")
        sys.exit()

    vessel_configs = [
        (1, "vessel_paths/1.json"),
        (2, "vessel_paths/2.json"),
        (3, "vessel_paths/3.json"),
    ]

    no_errors = True

    vessels = []
    for vessel_id, path in vessel_configs:
        print(f"Starting vessel {vessel_id}")
        proc = launch_vessel(vessel_id, path, ip, PORT, "logs")
        time.sleep(0.5)  # Give it a moment to initialize
        if proc.poll() is not None:
            console.print(
                f"[yellow]Warning:[/] vessel {vessel_id} exited early! "
                f"Check [dim underline]logs/vessel_{vessel_id}.log[/]"
            )
            no_errors = False
        else:
            vessels.append(proc)

    if no_errors:
        console.print("All processes launched successfully. Press Ctrl+C to stop.", style="bold green")
    else:
        console.print("Some processes launched with errors. Press Ctrl+C to stop.", style="bold yellow")
    try:
        while True:
            time.sleep(1)

            # Check if server died
            if server_process.poll() is not None:
                console.print("Server exited unexpectedly!", style="bold red")
                for p in vessels:
                    p.terminate()
                sys.exit()

            # Check each vessel
            for i, vessel_proc in enumerate(vessels):
                if vessel_proc.poll() is not None:
                    vessel_id = vessel_configs[i][0]
                    console.print(
                        f"[yellow]Warning:[/] vessel {vessel_id} disconnected! "
                        f"See [dim underline]logs/vessel_{vessel_id}.log[/]"
                    )
                    vessels[i] = None  # Mark as dead to avoid repeated warnings
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_process.terminate()
        for p in vessels:
            p.terminate()


main()
