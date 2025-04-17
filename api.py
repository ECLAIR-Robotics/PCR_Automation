import socket
import time

def send_message(message: str, host='localhost', port=3000):
    """Send a message to the ROS 2 TCP bridge node"""
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            print(f"Connected to {host}:{port}")
            sock.sendall(message.encode())
            print(f"Sent: {message}")

            # Optionally receive a response (1 KB max)
            response = sock.recv(1024).decode()
            print(f"Received: {response.strip()}")

    except ConnectionRefusedError:
        print(f"Could not connect to {host}:{port}. Is the container running?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        msg = input("Enter a message to send (or 'exit'): ")
        if msg.lower() == 'exit':
            break
        send_message(msg)
