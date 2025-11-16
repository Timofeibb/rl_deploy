cmake_minimum_required(VERSION 3.5)
project(a1_description VERSION 3.0.0)

find_package(catkin REQUIRED COMPONENTS
    genmsg
    roscpp
    std_msgs
    tf
)

catkin_package(
    CATKIN_DEPENDS
)

include_directories(
    # include
    ${catkin_INCLUDE_DIRS}
)
