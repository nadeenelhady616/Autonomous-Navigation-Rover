cmake_minimum_required(VERSION 3.8)
project(rover_description)

find_package(ament_cmake REQUIRED)

# Install directory assets (URDF, configs, worlds, launch files, Gazebo models)
install(
  DIRECTORY urdf config worlds launch rviz models
  DESTINATION share/${PROJECT_NAME}
)

# Install Python nodes as executable scripts so 'ros2 run' can locate them
install(
  PROGRAMS
    scripts/lane_controller.py
    scripts/lane.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()