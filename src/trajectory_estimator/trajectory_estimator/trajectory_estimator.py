import rclpy
from rclpy.node import Node

from abb_interface.srv import EEPose
from vicon_receiver.msg import Position as ViconPosition

import numpy as np
from typing import Union

class TrajectoryEstimator(Node):
  
  def __init__(self):
    super().__init__("trajectory_estimator")
    
    
    # MoveL client
    self.move_l_client = self.create_client(EEPose, "move_linear")
    self.req = EEPose.Request()
    
    # Vicon subscriber
    self.vicon_sub = self.create_subscription(ViconPosition, 
                                              "vicon/subject_1/segment_1", 
                                              self.vicon_callback,
                                              qos_profile = 10)
    
    # Trajectory estimation
    self.dt = 1./100.
    self.max_time = 2.0 # sec
    self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64) * 1000.0 # mm/s^2
    T = np.arange(0, self.max_time, self.dt, dtype=np.float64)
    self.times = np.column_stack([T, T, T])
    self.times_squared = self.times ** 2
    # ---------------------------------------
    self.prev_msg = None
    self.velocity_vector = None
    
    # Trajectory smoothing
    self.alpha = 0.8
    self.prev_trajectory : Union[None, np.ndarray]= None
  
  def send_request(self):
      # TODO: remove after service updates XYZ
      self.req.x = 0.
      self.req.y = -400.
      self.req.z = 500.
      
      self.req.qw = 0.707
      self.req.qx = 0.
      self.req.qy = 0.707
      self.req.qz = 0.
      
      self.req.lin_vel = 500.
      self.req.ang_vel = 50.      
      response = self.move_l_client.call(self.req)
      
  def vicon_callback(self, msg : ViconPosition):
    if self.prev_msg is not None:
      self.velocity_vector = np.array([
        msg.x_trans - self.prev_msg.x_trans,
        msg.y_trans - self.prev_msg.y_trans,
        msg.z_trans - self.prev_msg.z_trans
      ]) / self.dt # mm/s
      
      # 1. estimated trajectory using 'projectile motion'
      # 2. smooth estimated trajectory
      # 3. check if in robot workspace
      # 4. 
    
    self.prev_msg = msg
      
  def estimate_trajectory(self, r0, v0):
    return r0 + v0 * self.times + 0.5 * self.g * self.times_squared
  
  def exponential_smooth(self, new_trajectory):
    if self.prev_trajectory is not None:
      self.prev_trajectory = (1 - self.alpha) * self.prev_trajectory[1:]
    else:
      self.prev_trajectory = new_trajectory
      
def main(args = None):
  rclpy.init(args = args)
  
  node = TrajectoryEstimator()
  
  node.send_request()
  
  rclpy.spin_once(node)
  node.destroy_node()
  rclpy.shutdown()
    
  

if __name__ == "__main__":
  main()
