import socket
import threading
import rclpy
import json
import sys
from rclpy.node import Node
from std_msgs.msg import String
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import Call
from xarm_msgs.srv import SetInt16
from sensor_msgs.msg import JointState

#TODO: send JSON responses
#TODO: we return an error code when we actually run into issues
#TODO: Implement /ufactory_driver/set_parameters
#TODO: keep track of current parameters
#TODO: make sure to store the default parameters since parameters persist
#TODO: handle unknown cmd

class TCPBridgeNode(Node):
    def __init__(self):
        super().__init__('tcp_bridge_node')

        # ROS 2 publishers and subscribers
        self.pub = self.create_publisher(String, 'input_topic', 10)

        self.output_pub = self.create_publisher(String, 'output_topic', 10)
        self.sub = self.create_subscription(String, 
            'output_topic', 
            self.ros_to_socket_callback, 
            10)

        
        # Subscriber node that monitors for the robot state topic
        self.state_sub = self.create_subscription(RobotMsg, 
            '/ufactory/robot_states',
            self.robot_state_callback,
            10)

        # Subscriber node that monitors for the joint state topic
        self.joint_sub = self.create_subscription(JointState,
            '/ufactory/joint_states',
            self.joint_state_callback,
            10)
        
        self.robo_info = None #dict containing our robot state
        self.joint_info = None #dict containing our joint state


        # TCP server setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('', 3000))
        self.sock.listen(1)
        self.get_logger().info('TCPBridgeNode listening on port 3000...')

        # Accept connection in a separate thread
        threading.Thread(target=self.accept_connection, daemon=True).start()

    def accept_connection(self):
        self.conn, addr = self.sock.accept()
        self.get_logger().info(f'Client connected from {addr}')
        threading.Thread(target=self.listen_to_socket, daemon=True).start()

    """
    msg fmt:
    {
    cmd: Str
    args: Str[]
    }
    """


    def listen_to_socket(self):
        while rclpy.ok():
            try:
                data = self.conn.recv(1024)
                if not data:
                    if self.conn.fileno() == -1:
                        self.get_logger().info('Client disconnected')
                        break
                    else: continue # O.W try again
                message = data.decode().strip()
                input = json.loads(message)
                self.get_logger().info(f'Received over TCP: "{input}"')
                
                ros_msg = String()
                ros_msg.data = message
                self.pub.publish(ros_msg)

                #do something like evoke_serivce()

                # If message is 'home', trigger the service call
                self.call_service(input["cmd"],input["args"])
                        
            except Exception as e:
                self.get_logger().error(f"Socket receive error: {e}")
                break

    def call_service(self, cmd, args):
            #VISHAL
            if cmd == 'home':
                self.call_home_service()

            #VISHAL
            elif cmd == 'position':
                #evoke handle
                self.call_position_service()

            #VISHAL
            elif cmd == 'angle':
                self.call_angle_service()

            elif cmd == 'move':
                #parse "move x y z" fmt
                self.call_move_service(args)

            elif cmd == 'move_joint':
                self.call_joint_service(args)
            
            elif cmd == 'get_joint_velocity':
                self.call_get_joint_velocity_service()

            elif cmd == 'set_joint_velocity':
                self.call_set_joint_velocity_service(args)
            
            elif cmd == 'clean_error':
                self.call_clean_error()
            
            elif cmd == 'clean_warn':
                self.call_clean_warning()

            elif cmd == 'get_state':
                self.call_get_state()
                
            elif cmd == 'get_mode':
                self.call_get_mode()
            
            elif cmd == 'set_state':
                #only one arg
                self.call_set_state(*args)
            
            elif cmd == 'set_mode':
                #only one arg
                self.call_set_mode(*args)

    def call_set_mode(self,arg):
        #/ufactory/set_mode [xarm_msgs/srv/SetInt16]

        #create client
        client = self.create_client(SetInt16, '/ufactory/set_mode')

        #check and wait for the client for 5 seconds
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetInt16 service not avaliable')
            return
        
        #create request object to pass in input
        request = SetInt16.Request()
        request.data = int(arg[0])
        future = client.call_async(request)

        #subfunction that we will pass to callback()
        def on_result(future):
            ros_msg = String()
            ros_dict = {"cmd": "set_mode"}
            if future.result() is not None:
                self.get_logger().info(f'SetInt16 success: {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().ret
            else:
                self.get_logger().error('SetInt16 service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'set mode service call failed'
            
            ros_msg.data = json.dumps(ros_dict)

            self.output_pub.publish(ros_msg)

        future.add_done_callback(on_result)
    
    def call_set_state(self,arg):
        #/ufactory/set_state [xarm_msgs/srv/SetInt16]
        
        #create client
        client = self.create_client(SetInt16, '/ufactory/set_state')

        #check and wait for the client for 5 seconds
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetInt16 service not avaliable')
            return
        
        #create request object to pass in input
        request = SetInt16.Request()
        request.data = int(arg)
        future = client.call_async(request)

        #subfunction that we will pass to callback()
        def on_result(future):
            ros_msg = String()
            ros_dict = {"cmd": "set_state"}

            if future.result() is not None:
                self.get_logger().info(f'SetInt16 success: {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message
                
            else:
                self.get_logger().error('SetInt16 service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = future.result().message

            ros_msg.data = json.dumps(ros_dict)
            self.output_pub.publish(ros_msg)

        future.add_done_callback(on_result)
    
    def call_get_mode(self):
        if self.robo_info is None:
            self.get_logger().error('No data on mode avaliable')
            return
         
        if hasattr(self,'conn'):
            try:
                mode = self.robo_info["mode"]

                state_dict = {"cmd": "get_mode"}
                state_dict["mode"] = mode
                state_msg = json.dumps(state_dict)

                self.conn.sendall(state_msg.encode())
                self.get_logger().info('Sent mode data to client')
            except Exception as e: 
                self.get_logger().error(f'Failed to send mode data: {e}')
        
        return 
        
    
    def call_get_state(self):
        if self.robo_info is None:
            self.get_logger().error('No data on mode avaliable')
            return

        if hasattr(self,'conn'):
            try:
                state = self.robo_info["state"]

                state_dict = {"cmd": "get_state"}
                state_dict["state"] = state
                state_msg = json.dumps(state_dict)

                self.conn.sendall(state_msg.encode())
                self.get_logger().info('Sent state data to client')
            except Exception as e: 
                self.get_logger().error(f'Failed to send state data: {e}')
        
        return 
    
    def call_clean_error(self):

        client = self.create_client(Call, '/ufactory/clean_error')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('clean_error service not avaliable')
            return
        
        request = Call.Request()
        future = client.call_async(request)

        def on_result(future):
            ros_msg = String()
            ros_dict = {"cmd": "clean_error"}

            if future.result() is not None:
                self.get_logger().info(f'clean_error success: {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message
            else:
                self.get_logger().error('clean_error service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'clean_error service call failed'
                
            ros_msg.data = json.dumps(ros_dict)
            self.output_pub.publish(ros_msg)

        #when the service is completed, exec on_result()
        future.add_done_callback(on_result)
        
        return
    
    def call_clean_warning(self):
        
        client = self.create_client(Call, '/ufactory/clean_warn')
        
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('clean_error service not avaliable')
            return
        
        request = Call.Request()
        future = client.call_async(request)

        def on_result(future):
            ros_msg = String()
            ros_dict = {"cmd": "clean_warning"}

            if future.result() is not None:
                self.get_logger().info(f'clean_warn success: {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message
                
            else:
                self.get_logger().error('clean_warn service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'clean_warn service call failed'
            
            ros_msg.data = json.dumps(ros_dict)
            self.output_pub.publish(ros_msg)

        #when the service is completed, exec on_result()
        future.add_done_callback(on_result)
        
        return

    def call_home_service(self):
        from xarm_msgs.srv import MoveHome

        client = self.create_client(MoveHome, '/ufactory/move_gohome')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('MoveHome service not available')
            return

        request = MoveHome.Request()
        future = client.call_async(request)

        # NOTE: according to GPT this is leading to double spinning, ROS no likey
        # we already had ROS spinning from main(), no need for another thread
        #rclpy.spin_until_future_complete(self, future)

        #subfunction that we will pass to callback()
        def on_result(future):
            # create dictionary
            ros_dict = {"cmd": "home"}
            
            ros_msg = String()

            if future.result() is not None:
                self.get_logger().info(f'MoveHome success: {future.result().ret}')
                ros_dict['ret'] = future.result().ret
                ros_dict['message'] = future.result().message
            else:
                self.get_logger().error('MoveHome service call failed')
                ros_dict['ret'] = -1
                ros_dict['message'] = 'MoveHome service call failed'
                
            ros_msg.data = json.dumps(ros_dict)
            self.output_pub.publish(ros_msg)

        #when the service is completed, exec on_result()
        future.add_done_callback(on_result)

    def call_position_service(self):
        if self.robo_info is None:
            self.get_logger().error('No position data avaliable')
            return
        
        if hasattr(self,'conn'):
            try:
                position = self.robo_info["pose"][:3]
                rotation = self.robo_info["pose"][3:]
                
                state_dict = {"cmd": "position"}

                state_dict["position"] = position
                state_dict["rotation"] = rotation

                state_msg = json.dumps(state_dict)

                self.conn.sendall(state_msg.encode())
                self.get_logger().info('Sent position data to client')
            except Exception as e: 
                self.get_logger().error(f'Failed to send position data: {e}')
        
        return 

    def call_angle_service(self):
        if self.robo_info is None:
            self.get_logger().error('No position data avaliable')
            return
        
        if hasattr(self,'conn'):
            try:
                angle = self.robo_info["angle"]

                state_dict = {"cmd": "angle"}
                state_dict["angles": angle]

                state_msg = json.dumps(state_dict)

                self.conn.sendall(state_msg.encode())
                self.get_logger().info('Sent angle data to client')
            except Exception as e: 
                self.get_logger().error(f'Failed to send angle data: {e}')
        
        return
    
    def call_get_joint_velocity_service(self):
        if self.joint_info is None:
            self.get_logger().error('No joint velocity data avaliable')
            return
        
        if hasattr(self,'conn'):
            try:
                velocity = self.joint_info["velocity"]

                state_dict = {"cmd": "velocity"}
                state_dict["velocities"] = velocity
                state_msg = json.dumps(state_dict)

                self.conn.sendall(state_msg.encode())
                self.get_logger().info('Sent position data to client')
            except Exception as e: 
                self.get_logger().error(f'Failed to send position data: {e}')
        
        return 
    
    #NOTE: need to set to Mode 4
    def call_set_joint_velocity_service(self, args):
        from xarm_msgs.srv import MoveVelocity

        client = self.create_client(MoveVelocity, '/ufactory/vc_set_joint_velocity')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('vc_set_joint_velocity service not available')
            return

        try:
            velocities = list(map(float, args))
            request = MoveVelocity.Request()
            request.speeds = velocities  # Array of 6 joint velocities
            
            future = client.call_async(request)
            
        except ValueError as e:
            self.get_logger().error(f'Invalid velocity parameters: {e}')
        
        request = MoveVelocity.Request()
        request.speeds = velocities

        future = client.call_async(request)

        def on_result(future):
            ros_msg = String()

            ros_dict = {"cmd": "set_joint_velocity"}

            if future.result() is not None:
                self.get_logger().info(f'vc_set_joint_velocity success {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message

            else:
                self.get_logger().error('vc_set_joint_velocity service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'vc_set_joint_velocity service call failed'
                
            
            ros_msg.data = json.dumps(ros_dict)
            self.output_pub.publish(ros_msg)

        future.add_done_callback(on_result)

    def call_move_service(self,args):
        from xarm_msgs.srv import MoveCartesian

        client = self.create_client(MoveCartesian, '/ufactory/set_position')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('MoveCartesian service not available')
            return
        
        # numbers passed from json are supposed to floats
        try:
            x,y,z,speed,acc,mvtime = map(float,args) 
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'Invalid coordinates provided: {e}')
            if hasattr(self, 'conn'):
                error_msg = f"Error: Coordinates must be numbers\n"
                self.output_pub.publish(error_msg.encode())
        
        request = MoveCartesian.Request()
        request.pose = [x, y, z, 3.14, 0.0, 0.0]
        request.speed = speed
        request.acc = acc
        request.mvtime = mvtime

        future = client.call_async(request)

        def on_result(future):
            ros_msg = String()

            ros_dict = {"cmd": "move"}

            if future.result() is not None:
                self.get_logger().info(f'MoveCartesian success {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message
                
            else:
                self.get_logger().error('MoveCartesian service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'MoveCartesian service call failed'

            ros_msg.data = json.dumps(ros_dict)
            
            self.output_pub.publish(ros_msg)

        future.add_done_callback(on_result)

    def call_joint_service(self,args):
        from xarm_msgs.srv import MoveJoint

        client = self.create_client(MoveJoint, '/ufactory/set_servo_angle_j')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('set_servo_angle_j service not available')
            return
        
        try:
            *joint_angles, speed, acc, mvtime = map(float, args)
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'Invalid arguments provided: {e}')
            if hasattr(self, 'conn'):
                error_msg = f"Error: arguments must be numbers\n"
                self.output_pub.publish(error_msg.encode())
        
        request = MoveJoint.Request()
        request.angles = joint_angles
        request.speed = speed
        request.acc = acc
        request.mvtime = mvtime

        self.get_logger().info(f'speed {speed}')
        self.get_logger().info(f'speed {acc}')

        future = client.call_async(request)

        def on_result(future):
            ros_msg = String()
            ros_dict = {"cmd": "move_joint"}

            if future.result() is not None:
                self.get_logger().info(f'set_servo_angle_j success {future.result().ret}')
                ros_dict["ret"] = future.result().ret
                ros_dict["message"] = future.result().message
                
            else:
                self.get_logger().error('set_servo_angle_j service call failed')
                ros_dict["ret"] = -1
                ros_dict["message"] = 'set_servo_angle_j service call failed'
            
            ros_msg.data = json.dumps(ros_dict)
            
            self.output_pub.publish(ros_msg)

        future.add_done_callback(on_result)

    # Subscriber Actions

    def robot_state_callback(self, msg: RobotMsg):
        # update our values, only send when client requests them
        self.robo_info = {
            "state": msg.state,
            "mode": msg.mode,
            "cmdnum": msg.cmdnum,
            "mt_brake": msg.mt_brake,
            "mt_able": msg.mt_able,
            "err": msg.err,
            "warn": msg.warn,
            "angle": msg.angle,
            "pose": msg.pose,
            "offset": msg.offset
        }
    
    def joint_state_callback(self,msg: JointState):
        self.joint_info = {
            "name": msg.name,
            "position": msg.position,
            "velocity": msg.velocity,
            "effort": msg.effort
        }
        

    def ros_to_socket_callback(self, msg):
        self.get_logger().debug('Subscriber callback triggered')  # Add debug logging
        if hasattr(self, 'conn'):
            try:
                response = f"[ROS] {msg.data}\n"
                self.conn.sendall(response.encode())
                self.get_logger().info(f'Sent to TCP client: "{msg.data}"')
            except Exception as e:
                self.get_logger().error(f"Socket send error: {e}")

def cleanup(node):
    print("\nCleaning up...")
    
    # Close TCP connection if it exists
    if hasattr(node, 'conn') and node.conn:
        try:
            node.conn.close()
            print("TCP connection closed")
        except Exception as e:
            print(f"Error closing TCP connection: {e}")
    
    # Close server socket if it exists
    if hasattr(node, 'sock') and node.sock:
        try:
            node.sock.close()
            print("Server socket closed")
        except Exception as e:
            print(f"Error closing server socket: {e}")
    
    # Cleanup ROS node
    try:
        node.destroy_node()
        print("ROS node shut down")
    except Exception as e:
        print(f"Error during ROS shutdown: {e}")
    
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = TCPBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        cleanup(node)
        rclpy.shutdown()


if __name__ == '__main__':
    main()