FROM --platform=amd64 osrf/ros:humble-desktop

# will need to bind mount the working directory 

# only for documentation purposes, need to invoke upon calling run
EXPOSE 3000

RUN apt update && \
	apt install -y ros-humble-gazebo-* && \
	apt install -y ros-${ROS_DISTRO}-moveit