"""One-command bring-up: Gazebo + rover + lane vision + lane/lidar controller
+ lidar costmap + RViz.

    ros2 launch rover_description autonomous.launch.py
    ros2 launch rover_description autonomous.launch.py rviz:=false
    ros2 launch rover_description autonomous.launch.py drive:=false   # vision/costmap only, rover stays still
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('rover_description')

    costmap_params = os.path.join(pkg, 'config', 'costmap.yaml')
    rviz_config = os.path.join(pkg, 'rviz', 'rover.rviz')

    args = [
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with costmap, lidar and lane views'),
        DeclareLaunchArgument('drive', default_value='true',
                              description='Start lane_controller (false = rover will not move)'),
        DeclareLaunchArgument('start_delay', default_value='10.0',
                              description='Seconds to wait for Gazebo + controllers before '
                                          'starting lane nodes / costmap'),
    ]

    # 1) Gazebo world + rover + robot_state_publisher + bridges + ros2_control controllers
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'spawn_robot.launch.py')
        )
    )

    # 2) Vision: camera -> white-line / obstacle detection -> /lane/*
    lane_detector = Node(
        package='rover_description',
        executable='lane.py',
        name='lane_detector_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # 3) Driving: /lane/* + /scan -> /cmd_vel (TwistStamped)
    lane_controller = Node(
        package='rover_description',
        executable='lane_controller.py',
        name='lane_controller_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('drive')),
    )

    # 4) Lidar costmap (standalone Nav2 costmap_2d, rolling window around the rover)
    local_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[costmap_params, {'use_sim_time': True}],
    )

    # The costmap is a lifecycle node: it does nothing until configured + activated.
    costmap_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_costmap',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['local_costmap'],
            'bond_timeout': 0.0,   # standalone costmap doesn't create a bond
        }],
    )

    delayed = TimerAction(
        period=LaunchConfiguration('start_delay'),
        actions=[local_costmap, costmap_lifecycle, lane_detector, lane_controller],
    )

    # 5) RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(args + [sim, rviz, delayed])
