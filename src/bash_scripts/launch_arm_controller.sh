#!/bin/bash

# Run the 'sws' alias
sws

# Launch the ROS2 node
ros2 launch xarm_api lite6_driver.launch.py robot_ip:=192.168.1.166 report_type:=normal
