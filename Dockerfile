FROM --platform=amd64 osrf/ros:humble-desktop

# Documentation port
EXPOSE 3000

# Install necessary ROS tools and packages
RUN apt update && \
    apt install -y \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-diagnostic-updater* \
        ros-humble-control-msgs* \
        ros-humble-librealsense2* \
        ros-humble-hardware-interface* \
        ros-humble-controller-manager* \
        ros-humble-moveit \
        ros-humble-moveit-* \
        ros-humble-ros-gz \
		netcat \
        ros-humble-gazebo-* && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || echo "rosdep already initialized" && \
    rosdep update

# Set default entrypoint to bash
CMD ["bash"]
