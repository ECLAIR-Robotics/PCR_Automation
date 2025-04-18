import socket
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TCPBridgeNode(Node):
    def __init__(self):
        super().__init__('tcp_bridge_node')

        # ROS 2 publishers and subscribers
        self.pub = self.create_publisher(String, 'input_topic', 10)
        self.sub = self.create_subscription(String, 'output_topic', self.ros_to_socket_callback, 10)

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

    def listen_to_socket(self):
        while rclpy.ok():
            try:
                data = self.conn.recv(1024)
                if not data:
                    self.get_logger().info('Client disconnected')
                    break
                message = data.decode().strip()
                self.get_logger().info(f'Received over TCP: "{message}"')
                ros_msg = String()
                ros_msg.data = message
                self.pub.publish(ros_msg)

                # If message is 'home', trigger the service call
                if message.lower() == 'home':
                    self.call_home_service()

            except Exception as e:
                self.get_logger().error(f"Socket receive error: {e}")
                break


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
            if future.result() is not None:
                self.get_logger().info(f'MoveHome success: {future.result().ret}')
            else:
                self.get_logger().error('MoveHome service call failed')
        
        #when the service is completed, exec on_result()
        future.add_done_callback(on_result)


    def ros_to_socket_callback(self, msg):
        if hasattr(self, 'conn'):
            try:
                response = f"[ROS] {msg.data}\n"
                self.conn.sendall(response.encode())
                self.get_logger().info(f'Sent to TCP client: "{msg.data}"')
            except Exception as e:
                self.get_logger().error(f"Socket send error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TCPBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()