#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R

class StaticTransformBroadcaster(Node):

    def __init__(self):
        super().__init__('static_transform_broadcaster')
        self.publisher = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transform()

    def publish_static_transform(self):
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'tof'
        transform.child_frame_id = 'camera_link'

        transform.transform.translation.x = 1.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
        quat = rot.as_quat()  # Roll, pitch, yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.get_logger().info('Publishing static transform from base_link to camera_link')

        self.publisher.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    
    node = StaticTransformBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
