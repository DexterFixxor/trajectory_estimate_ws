import launch
import launch_ros.actions
from launch_ros.actions.node import Node

def generate_launch_description():
    return launch.LaunchDescription([
        Node(
            package='abb_communication',
            executable='communication',
            name='abb_communication',
            parameters=[
              {
                "ip": "192.168.125.1"  
              }
            ],
            output="screen",
            emulate_tty=True,
            ),
  ])
