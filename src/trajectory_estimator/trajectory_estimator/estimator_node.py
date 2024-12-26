import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import Point
from vicon_receiver.msg import Position

from abb_interface.srv import EEPose
from .estimator import Estimator


import numpy as np
from typing import Union

class TrajectoryEstimator(Node):
  
  def __init__(self):
    super().__init__("trajectory_estimator")
    # MoveL client
    self.move_l_client = self.create_client(EEPose, "move_linear")
    self.req = EEPose.Request()
    
    # Vicon subscriber
    self.vicon_sub = self.create_subscription(Position, 
                                              "vicon/D12/D12", 
                                              self.vicon_callback,
                                              qos_profile = 10)
    
    # Trajectory estimation
    self.estimator = Estimator(dt = 1 / 100., 
                               max_time = 1.0,
                               xyz_min = np.array([-150, -100, 150], dtype=np.float32),
                               xyz_max = np.array([220, 500, 700], dtype=np.float32),
                               alpha = 0.8 # exponential smoothing
                               )
    # ---------------------------------------
    print("\nStarting Trajectory Estimator node...\n\n")

  def send_request(self, coords : np.ndarray):
      self.req.x = coords[0]
      self.req.y = coords[1]
      self.req.z = coords[2]
      
      self.req.qw = 0.0
      self.req.qx = 0.0
      self.req.qy = 1.0
      self.req.qz = 0.0
      
      self.req.lin_vel = 1000.
      self.req.ang_vel = 200.      
      future_ret = self.move_l_client.call_async(self.req)
      future_ret.add_done_callback(self.service_future_callback)
  
  def service_future_callback(self, future):
      pass
    
  def vicon_callback(self, msg : Position):
      
      np_msg = np.array([msg.x_trans, msg.y_trans, msg.z_trans], dtype=np.float32)
      try:
        self.estimator.position_callback(np_msg)
        
      except RuntimeError as e:
        print("Error in estimation: ", e)    
        pass
      # if DKL converged send request to MoveL service
      if self.estimator.flg_done:
        self.send_request(self.estimator.saved_coord)
        print("\n", self.estimator.saved_coord)
        self.estimator.reset()
        
        
      
def main(args = None):
  rclpy.init(args = args)
  
  node = TrajectoryEstimator()
  rclpy.spin(node)
  
  node.destroy_node()
  rclpy.shutdown()
    
if __name__ == "__main__":
  main()
