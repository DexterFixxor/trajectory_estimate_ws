import rclpy
import rclpy.logging
from rclpy.node import Node

import abb_communication.open_abb as open_abb
from abb_interface.srv import EEPose
from std_srvs.srv import Empty

class CommunicationNode(Node):
  
  def __init__(self):
    super().__init__("abb_communication")
    
    self.declare_parameter(name="ip", value="192.168.125.1")
    ip = self.get_parameter("ip").get_parameter_value().string_value
    
    print()
    print("*"*80)
    print(f"Connecting to controller with IP address: {ip}")
    print("*"*80)
    self.robot = open_abb.Robot(ip=ip)
    
    self.robot.set_workobject([
      [579.715, -201.8785, 75.48726], 
      [1, 0, 0, 0]
    ])
    
    self.robot.set_tool(
      [
        [-70, 0, 70],
        [1.0, 0.0, 0.0, 0.0]
      ]
    )
    
    print('*'*50)
    print()
    print("Connected to ABB robot.")
    print("Creating ABB robot 'move' service...")
    print('.\n.\n.')
    
    self.move_l_service = self.create_service(EEPose, "move_linear", self.move_l_callback)
    self.home_service = self.create_service(Empty, "move_home", self.move_home_callback)    
    print("Service created...")
    print('*'*50)
    

  def move_l_callback(self, request : EEPose.Request, response : EEPose.Response):
    print("Got request")
    pos = [
      request.x,
      request.y,
      request.z
    ]
    
    rot = [
      request.qw,
      request.qx,
      request.qy,
      request.qz
    ]
    if request.lin_vel != 0.0 and request.ang_vel != 0.0:
      self.robot.set_speed([
        request.lin_vel,
        request.ang_vel,
        50,
        50
      ])
    
    self.robot.set_cartesian([pos, rot])
    
    response.is_ok = True
    return response
  
  def move_home_callback(self, request, response):
    self.robot.set_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return response

  def shutdown(self):
    print("Shutting down ABB communication node.\n")
    self.robot.close()

def main(args = None):
  rclpy.init(args = args)

  node = CommunicationNode()
  
  node.context.on_shutdown(node.shutdown)
  
  rclpy.spin(node)
  
  node.destroy_node()
  rclpy.shutdown()

if __name__ == "__main__":
  main()
