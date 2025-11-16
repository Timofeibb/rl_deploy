cmake_minimum_required(VERSION 3.5)
project(robot_joint_controller VERSION 3.0.0 LANGUAGES CXX)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  hardware_interface
  pluginlib
  roscpp
  realtime_tools
  robot_msgs
)

catkin_package(
  CATKIN_DEPENDS
    robot_msgs
    controller_interface
    hardware_interface
    pluginlib
    roscpp
    realtime_tools
  INCLUDE_DIRS include
  LIBRARIES robot_joint_controller
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(robot_joint_controller SHARED
  src/robot_joint_controller.cpp
)

target_link_libraries(robot_joint_controller
  ${catkin_LIBRARIES}
)

# install rules
install(TARGETS robot_joint_controller
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES robot_joint_controller_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)