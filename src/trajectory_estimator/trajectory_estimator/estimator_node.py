import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from abb_interface.srv import EEPose
from estimator import Estimator


import numpy as np
from typing import Union

class TrajectoryEstimator(Node):
  
  def __init__(self):
    super().__init__("trajectory_estimator")
    # MoveL client
    self.move_l_client = self.create_client(EEPose, "move_linear")
    self.req = EEPose.Request()
    
    # Vicon subscriber
    self.vicon_sub = self.create_subscription(Point, 
                                              "vicon/subject_1/segment_1", 
                                              self.vicon_callback,
                                              qos_profile = 10)
    
    # Trajectory estimation
    self.estimator = Estimator(dt = 1 / 100., 
                               max_time = 1.0,
                               xyz_min = np.array([0, 0, 0], dtype=np.float32),
                               xyz_max = np.array([400, 400, 400], dtype=np.float32),
                               alpha = 0.8 # exponential smoothing
                               )
    # ---------------------------------------

  def send_request(self, coords : np.ndarray):
      self.req.x = coords[0]
      self.req.y = coords[1]
      self.req.z = coords[2]
      
      self.req.qw = 0.707
      self.req.qx = 0.
      self.req.qy = 0.707
      self.req.qz = 0.
      
      self.req.lin_vel = 500.
      self.req.ang_vel = 50.      
      future_ret = self.move_l_client.call_async(self.req)
      future_ret.add_done_callback(self.service_future_callback)
  
  def service_future_callback(self, future):
      pass
    
  def vicon_callback(self, msg : Point):
      
      np_msg = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
      self.estimator.position_callback(np_msg)
      
      # if DKL converged send request to MoveL service
      if self.estimator.flg_done:
        self.send_request(self.estimator.saved_coord)
        self.estimator.reset()
        
        
      
def main(args = None):
  rclpy.init(args = args)
  
  node = TrajectoryEstimator()
  rclpy.spin(node)
  
  node.destroy_node()
  rclpy.shutdown()
    
if __name__ == "__main__":
  main()
