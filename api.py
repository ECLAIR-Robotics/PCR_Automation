import socket
import time
import json

#TODO: Take in JSON responses
#TODO: have error responses be more fleshed out

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
            response = self.sock.recv(1024).decode().strip()
            try:
                output = json.loads(response)
                print(f"Received: {response}")
                return output
            except json.JSONDecodeError:
                print(f"Received: {response}")
                return {"response": response}
            #return output?

        #raise any exception and remove socket
        except Exception as e:
            print(f"Error: {e}")
            self.sock = None #reconnect on next message
            raise

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


#NOTE: add more commands with arguments
commands = {
    "move": {"args": ["x","y","z","speed","acc","mvtime"]}, #input: float32[] x,y,z
    "home": {"args": []},
    "position": {"args": []}, 
    "angle": {"args": []},
    "move_joint": {"args": ["j1","j2","j3","j4","j5","j6","speed","acc","mvtime"]}, #input: float32[] angles j1-j6
    "get_joint_velocity": {"args": []},
    "set_joint_velocity": {"args": ["j1","j2","j3","j4","j5","j6"]},
    "clean_error": {"args":[]},
    "clean_warn": {"args":[]},
    "get_state": {"args":[]},
    "get_mode": {"args":[]},
    "set_state": {"args":["state"]}, #input: int16
    "set_mode": {"args":["state"]}, #input: int16
}

if __name__ == "__main__":
    client = ROSBridgeClient()

    try:
        client.connect()
        while True:
            userInput = input("Enter a message to send ('help', or 'exit'): ")
            if userInput.lower() == 'exit' or userInput == "":
                break
            elif userInput.lower() == 'help':
                print(commands)

            #split the string 
            split = userInput.split(" ")

            #1st arg is the cmd name
            cmd = split[0]

            if cmd not in commands:
                print(f"Unknown command: {cmd}")
                continue
            
            input_data  = {"cmd": cmd}

            #take in the required args as list
            required_args = commands[cmd]["args"]

            # check to see if the length vs args list
            if len(split)-1 != len(required_args):
                print(f"Command {cmd} requires {len(required_args)} arguments: {required_args}")
                continue
            
            input_data["args"] = split[1:] if len(split) > 1 else []
            
            #any other args based on " "
            msg = json.dumps(input_data)
            client.send_message(msg)
    finally:
        client.close()
