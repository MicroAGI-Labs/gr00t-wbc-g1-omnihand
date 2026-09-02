import os
import launch
import xacro
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    # package
    pkg_dir = get_package_share_directory('omnihand_description')

    xacro_path = os.path.join(get_package_share_directory(
        'omnihand_description'), 'urdf/xacro', 'omnihand_right.xacro')
    
    rviz_arguments = ['-d', os.path.join(pkg_dir, 'rviz', 'omnihand_right.rviz')]

    hand_type = LaunchConfiguration('hand_type').perform(context)  
    # urdf
    if hand_type == 'left':
        xacro_path = os.path.join(get_package_share_directory(
        'omnihand_description'), 'urdf/xacro', 'omnihand_left.xacro')
        rviz_arguments=['-d', [os.path.join(pkg_dir, 'rviz', 'omnihand_left.rviz')]]
    
    urdf_contents = xacro.process_file(xacro_path).toprettyxml(indent='  ')
    # urdf_file = os.path.join(pkg_dir, 'urdf', 'omnihand.urdf')
    # with open(urdf_file, 'r') as infp:
    #     urdf_contents = infp.read()

    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[
            {'robot_description': urdf_contents}
        ]
    )
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': urdf_contents},
            {'publish_frequency': 500.0}
        ]
    )
    
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_arguments
    )

    # rviz2_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     output='screen'
    # )

    return [joint_state_publisher_node, robot_state_publisher_node, rviz2_node]

def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            name='hand_type',
            default_value='right',  # Default is 'right', can be overridden in launch command
            description='Hand type: right or left'
        ),
        OpaqueFunction(function=launch_setup)
    ])