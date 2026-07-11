import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('scan_lock')
    default_config_path = os.path.join(pkg_dir, 'config')
    default_rviz_config = os.path.join(pkg_dir, 'rviz', 'pose_prompt.rviz')

    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='false'))
    ld.add_action(DeclareLaunchArgument(
        'config_path', default_value=default_config_path))
    ld.add_action(DeclareLaunchArgument(
        'config_file', default_value='pose_prompt.yaml'))
    ld.add_action(DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config,
        description='RViz config file path'))

    ld.add_action(Node(
        package='scan_lock',
        executable='pose_prompt_node',
        name='pose_prompt',
        parameters=[
            PathJoinSubstitution([config_path, config_file]),
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    ))

    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='pose_prompt_rviz',
        arguments=['-d', rviz_cfg],
        output='screen',
    ))

    return ld
