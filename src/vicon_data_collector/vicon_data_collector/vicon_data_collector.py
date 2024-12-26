import rclpy
from rclpy.node import Node
import rclpy.qos
from vicon_receiver.msg import Position
import numpy as np
import cv2

class DataCollectorNode(Node):
    
    def __init__(self):
        
        super().__init__("vicon_data_collector")
        self.vicon_sub = self.create_subscription(
            Position,
            "vicon/D12/D12",
            self.vicon_callback,
            100
        )
        
        self.flg_record = False
        self.data_list = []
        self.data_cnt_index = 0

    def vicon_callback(self, msg : Position):
        if self.flg_record:
            
            xyz = [msg.x_trans, msg.y_trans, msg.z_trans]
            self.data_list.append(xyz)
            

def main():
    
    rclpy.init()
    node = DataCollectorNode()
    
    dummy_img = np.zeros((4,4))
    cv2.imshow("frame", dummy_img)
    
    while rclpy.ok():
        rclpy.spin_once(node)
        
        if cv2.waitKey(1) == 114: # 'r'
            if not node.flg_record:
                print("Starting to record trajectory...")
                node.flg_record = True
            else:
                print(f"Saving trajectory: trajectory_{node.data_cnt_index:04d}")
                
                data_path = f"/home/dexter/Programming/RoboticsFTN/balltrack_ws/src/vicon_data_collector/data/trajectory_{node.data_cnt_index:04d}"
                np.save(data_path, np.array(node.data_list))
                
                node.data_cnt_index += 1
                node.flg_record = False
                node.data_list = []

                print("-" * 50)
    
    node.destroy_node()
    
    rclpy.shutdown()
        
    

if __name__ == '__main__':
    main()
