import socket
import time

class ROSBridgeClient:
    def __init__(self, host='localhost', port=3000):
        self.host = host
        self.port = port
        self.sock = None
    
    def connect(self):

        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5)
            print(f"Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            print(f"Could not connect to {self.host}:{self.port}. Is the container running?")
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def send_message(self, message: str):
        if not self.sock:
            self.connect()

        #try sending a message
        try:
            self.sock.sendall(message.encode())
            print(f"Sent: {message}")
            
            # Optionally receive a response (1 KB max)
            response = self.sock.recv(1024).decode()
            print(f"Received: {response.strip()}")

        #raise any exception and remove socket
        except Exception as e:
            print(f"Error: {e}")
            self.sock = None #reconnect on next message
            raise

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


if __name__ == "__main__":
    client = ROSBridgeClient()
    try:
        client.connect()
        while True:
            msg = input("Enter a message to send (or 'exit'): ")
            if msg.lower() == 'exit':
                break
            client.send_message(msg)
    finally:
        client.close()
