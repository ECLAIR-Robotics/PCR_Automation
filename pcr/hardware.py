import serial 
import time 

arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1) 
time.sleep(2)

def write_read(x): 
	arduino.write(bytes(x, 'utf-8')) 
	time.sleep(0.05) 
	data = arduino.readline() 
	return data 

def run_command(cmd):
    arduino.write(f"{cmd}\n".encode())  # Send 'get\n' or 'eject\n'
    print(f"Sent '{cmd}' to Arduino.")
    while True:
        response = arduino.readline().decode().strip()
        if response:
            print(f"[Arduino]: {response}")
        if response == "done":
            print(f"{cmd.capitalize()} motion complete.")
            break
        
def get_liquid():
	run_command("get")

def eject_liquid():
    run_command("eject")
    
eject_liquid()
