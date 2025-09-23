import rclpy
import rclpy.logging
from rclpy.node import Node
# from geometry_msgs.msg import Point
from vicon_receiver.msg import Position

from abb_interface.srv import EEPose
from .estimator import Estimator


import numpy as np
from typing import Union

from utils import construct_new_frame

class TrajectoryEstimator(Node):
  
  def __init__(self):
    super().__init__("trajectory_estimator")
    # MoveL client
    self.move_l_client = self.create_client(EEPose, "move_linear")
    self.req = EEPose.Request()
    
    # Vicon subscriber
    self.vicon_sub = self.create_subscription(Position, 
                                              "vicon/ball/seg", 
                                              self.vicon_callback,
                                              qos_profile = 10)
    
    
    p_origin = np.array([])
    p_x_dir = np.array([])
    p_y_dir = np.array([])
    

    
    self.xyz_min = [20.0, 100.0, 150]
    self.xyz_max = [220, 500, 300]
    
    
    # Trajectory estimation
    self.estimator = Estimator(
          dt = 1/100.0,
          xyz_min = self.xyz_min,
          xyz_max = self.xyz_max,
          alpha = 0.9,
          dkl_th = 0.05,
          keep_last_gauss = 100,
          flg_use_nn = True
      )
    
    print("\n################### Estimator CONFIG ###################\n")
    print(self.estimator)
    print("\n########################################################\n")
    
    # ---------------------------------------
    print("\nStarting Trajectory Estimator node...\n\n")

  def send_request(self, coords : np.ndarray):
      
      self.req.x = float(coords[0])
      self.req.y = float(coords[1])
      self.req.z = float(coords[2])
      
      self.req.qw = 0.707106781
      self.req.qx = 0.0
      self.req.qy = 0.707106781
      self.req.qz = 0.0
      
      # self.req.qw = 0.5
      # self.req.qx = 0.0
      # self.req.qy = 0.866025404
      # self.req.qz = 0.0
      
      #self.req.lin_vel = 1000.
      #self.req.ang_vel = 200.      
      # self.get_logger().info(f"Sending request: {coords}")
      
      future_ret = self.move_l_client.call_async(self.req)
      future_ret.add_done_callback(self.service_future_callback)
      
  
  def service_future_callback(self, future):
      # self.get_logger().info("Service done")
      pass
    
  def vicon_callback(self, msg : Position):
      np_msg = np.array([msg.x_trans, msg.y_trans, msg.z_trans], dtype=np.float32)
      prev_has_converged = self.estimator.has_converged()
      try:
        self.estimator.position_callback(np_msg)
        
      except RuntimeError as e:
        print("Error in estimation: ", e)    
        
      # if DKL converged send request to MoveL service
      if self.estimator.has_converged():
        self.send_request(self.estimator.saved_coord)
        self.estimator.gauss.has_converged = False
        if not prev_has_converged:
          self.estimator.reset()
        
        
      
def main(args = None):
  rclpy.init(args = args)
  
  node = TrajectoryEstimator()
  rclpy.spin(node)
  
  node.destroy_node()
  rclpy.shutdown()
    
if __name__ == "__main__":
  main()
