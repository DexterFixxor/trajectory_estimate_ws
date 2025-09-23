import rclpy
import rclpy.logging
from rclpy.node import Node
# from geometry_msgs.msg import Point
from vicon_receiver.msg import Position

from abb_interface.srv import EEPose
from std_srvs.srv import Empty

import numpy as np


class WandTracker(Node):
  
    def __init__(self):
        super().__init__("trajectory_estimator")
        # MoveL client
        self.move_l_client = self.create_client(EEPose, "move_linear")
        self.req = EEPose.Request()
        
        self.move_home = self.create_client(Empty, "move_home")
        
          # Vicon     subscriber
        self.vicon_su    = self.create_subscription(Position, 
                                                  "vicon/Wand/seg", 
                                                  self.vicon_callback,
                                                  qos_profile = 10)
    

        self.flg_home = True
        
    def vicon_callback(self, msg : Position):
        
        if self.flg_home and msg.z_trans > 700:
            
            self.flg_home = False
            
            self.req.qw = 0.707106781
            self.req.qx = 0.0
            self.req.qy = 0.707106781
            self.req.qz = 0.0
            
            x_add = np.random.uniform(low = -20, high = 20)
            y_add = np.random.uniform(low = -30, high = 100)
            z_add = np.random.uniform(low = -100, high = 50)
            
            self.req.x = 150.0 + x_add
            self.req.y = 0.0 + y_add
            self.req.z = 500.0 + z_add
            
            future_ret = self.move_l_client.call_async(self.req)
            future_ret.add_done_callback(self.service_future_callback)
        # elif not self.flg_home:
        #     self.flg_home = True
        #     future_ret = self.move_home.call_async(Empty.Request())
        #     future_ret.add_done_callback(self.service_future_callback)
            
      
    def service_future_callback(self, future):
        future_ret = self.move_home.call_async(Empty.Request())
        future_ret.add_done_callback(self.service_home_future_callback)
      
    def service_home_future_callback(self, future):
        self.get_logger().info("\nGOT HOME!\n")
        self.flg_home = True

    
def main(args = None):
    rclpy.init(args = args)
  
    node = WandTracker()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
