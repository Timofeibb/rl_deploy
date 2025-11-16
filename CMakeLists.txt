cmake_minimum_required(VERSION 3.0.2)
project(rl_deploy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  gazebo_msgs
  controller_manager
  joint_state_controller
  position_controllers
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    sensor_msgs 
    geometry_msgs
    gazebo_msgs
    controller_manager
)

catkin_install_python(PROGRAMS
  scripts/rl_policy_node.py
  scripts/observation_manager.py
  scripts/fsm_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
