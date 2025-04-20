src: 
	/bin/bash -c "source install/setup.bash"

colcon:
	colcon build --executor sequential

run-arm:
	bash bash_scripts/launch_arm_controller.sh

run-tcp:
	ros2 run tcp_bridge_pkg tcp_bridge_node