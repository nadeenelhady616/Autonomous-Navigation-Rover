import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'rover_description'
    xacro_file_path = 'urdf/turtlebot3_burger.xacro'

    xacro_file = os.path.join(
        get_package_share_directory(pkg_name), 
        xacro_file_path
    )


    doc = xacro.process_file(xacro_file)
    robot_description_xml = doc.toprettyxml(indent='  ')


    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_description_xml,
            '-name', 'rover',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.2' 
        ],
        output='screen'
    )


    return LaunchDescription([
        spawn_robot_node
    ])