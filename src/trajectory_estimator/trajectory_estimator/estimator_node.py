import rclpy
from rclpy.node import Node
from vicon_receiver.msg import Position
import numpy as np

from .estimator import Estimator
from .utils import construct_new_frame
from xarm.wrapper import XArmAPI

class TrajectoryEstimator(Node):
  
  def __init__(self):
    super().__init__("trajectory_estimator")
    
    
    # xArm robot init
    self.ip = '10.1.108.144'
    self.arm = XArmAPI(self.ip)
    self.arm.motion_enable(enable=True)
    self.arm.set_mode(1)
    self.arm.set_state(state=0)
    self.arm_speed = 50
    
    # Vicon subscriber
    self.vicon_sub = self.create_subscription(Position, 
                                              "vicon/PurpleBall_10Markers/ball", 
                                              self.vicon_callback,
                                              qos_profile = 10)
    
    
    p_origin = np.array([301.4, -67.6, -124.6])
    p_x_dir = np.array([461.6, -68.6, -121.6])
    p_y_dir = np.array([305.2, 180.0, -123.2])
    
    self.T_base_vicon = construct_new_frame(p_origin, p_x_dir, p_y_dir)
    
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
    self.counter = 0
    print("\n################### Estimator CONFIG ###################\n")
    print(self.estimator)
    print("\n########################################################\n")
    
    # ---------------------------------------
    print("\nStarting Trajectory Estimator node...\n\n")
    
  def vicon_callback(self, msg : Position):
      np_msg = np.array([msg.x_trans, msg.y_trans, msg.z_trans], dtype=np.float32)

      try:
        self.estimator.position_callback(np_msg)
        
      except RuntimeError as e:
        print("Error in estimation: ", e)    
        
      self.counter += 1
      
      # if msg.z_trans > 150:
      #   target_pose = self.T_base_vicon @ np.array([0.0, 0.0, 1.0, 1], dtype=np.float32)
      #   self.arm.set_position(*target_pose[:3].tolist(), speed = 50)
      # if (- 100 < msg.x_trans < 280) and \
      #    (- 150 < msg.y_trans < 260) and \
      #    ( 150 < msg.z_trans < 400):# and (self.counter % 25 == 0):
           
      #   target_pose = self.T_base_vicon @ np.array([msg.x_trans, msg.y_trans, msg.z_trans, 1], dtype=np.float32)
        
      #   mvpose = [target_pose[0], target_pose[1], target_pose[2] + 40, 180, 0, 0]
      #   self.arm.set_servo_cartesian(mvpose, speed = 50, mvacc = 1000)
        
      
def main(args = None):
  rclpy.init(args = args)
  
  node = TrajectoryEstimator()
  rclpy.spin(node)
  
  node.destroy_node()
  rclpy.shutdown()
    
if __name__ == "__main__":
  main()
