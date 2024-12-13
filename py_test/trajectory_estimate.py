import csv
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection    


class Estimator:
    
    def __init__(self, dt, max_time, xyz_min, xyz_max):
        self.dt = dt
        self.max_time = max_time
        
        T = np.arange(0, self.max_time, self.dt, dtype=np.float64)
        self.times = np.column_stack([T, T, T])
        self.times_squared = self.times ** 2
        self.g = np.array([0.0, 0.0, -9.81]) * 1000 # mm/s^2

        self.prev_pos = None
        self.velocity = None
        self.prev_trajectory = None
        self.alpha = 0.7
        self.smooth_time = 0.9 * self.max_time
        self.index_smooth_time = int(self.smooth_time / dt)
        
        # workspace check
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        
    def position_callback(self, pos : np.ndarray):
        trajectory = None
        in_workspace = None
        if self.prev_pos is not None:
            self.velocity = (pos - self.prev_pos) / self.dt
        
            # ---------- BEGIN: estimate trajectory
            trajectory = self.estimate_trajectory()
            # self.prev_trajectory = trajectory
            # ---------- END: estimate trajectory
            
            # ---------- BEGIN: smooth trajectory
            self.smooth_trajectory(trajectory)
            # ---------- END: smooth trajectory
                        
            # ---------- BEGIN: check if in workspace
            in_workspace = (self.prev_trajectory >= self.xyz_min) & (self.prev_trajectory <= self.xyz_max)
            in_workspace = np.all(in_workspace, axis=1)
            if np.any(in_workspace):
                last_coords = self.prev_trajectory[in_workspace][-1]
                print(last_coords)            
            # ---------- END: check if in workspace
        self.prev_pos = pos
        
        return trajectory, in_workspace
            
    def estimate_trajectory(self):
        if self.prev_pos is not None and self.velocity is not None:
            return self.prev_pos + self.velocity * self.times + 0.5 * self.g * self.times_squared
        raise ValueError("previous_pos and velocity can not be 'None'. Function called before those parameters are calcualted")
    
    def smooth_trajectory(self, new_trajectory : np.ndarray):
        if self.prev_trajectory is not None:
            
            tmp = self.prev_trajectory[1]
            self.prev_trajectory[-self.index_smooth_time:] = self.alpha * new_trajectory[:self.index_smooth_time] + (1 - self.alpha) * self.prev_trajectory[1:self.index_smooth_time+1]
            self.prev_trajectory[0] = tmp
        else:
            self.prev_trajectory = new_trajectory[:self.index_smooth_time+1]

def read_vicon_file(file_path : str):
    fieldnames=['Frame','Sub Frame','RX','RY','RZ','TX','TY','TZ']
    file = open(data_path)
    csv_file = csv.reader(file)
    
    sorted_data = []
    
    flg_start = False
    for i, row in enumerate(csv_file):
        
        if flg_start:
            try:
                frame_data = {
                    "frame": int(row[0]),
                    "RX": float(row[2]),
                    "RY": float(row[3]),
                    "RZ": float(row[4]),
                    "X": float(row[-3]),
                    "Y": float(row[-2]),
                    "Z": float(row[-1])
                }
                
                sorted_data.append(frame_data)
                
            except:
         
                print(f"SKIPPING LINE: {i}", row)
                if len(row) > 0 and row[0] == "Trajectories":
                    flg_start = False
                    print("*"*50)
                    print("Process complete...")
                    print("*"*50)
                    
        if row == fieldnames:
            flg_start = True
            
            
    tx = [d["X"] for d in sorted_data]
    ty = [d["Y"] for d in sorted_data]
    tz = [d["Z"] for d in sorted_data]       
    
    rx = [d["RX"] for d in sorted_data]
    ry = [d["RY"] for d in sorted_data]
    rz = [d["RZ"] for d in sorted_data]       
    
    
    return np.column_stack([rx, ry, rz]), np.column_stack([tx, ty, tz])
 

def load_all_npy(folder_path):
    return glob.glob(os.path.join(folder_path, "**/*.npy"), recursive=True)

def plot_bbox(ax, xyz_min, xyz_max):
    # Create the list of vertices of the bounding box
    vertices = [
        [xyz_min[0], xyz_min[1], xyz_min[2]],  # (x1, y1, z1)
        [xyz_min[0], xyz_min[1], xyz_max[2]],  # (x1, y1, z2)
        [xyz_min[0], xyz_max[1], xyz_min[2]],  # (x1, y2, z1)
        [xyz_min[0], xyz_max[1], xyz_max[2]],  # (x1, y2, z2)
        [xyz_max[0], xyz_min[1], xyz_min[2]],  # (x2, y1, z1)
        [xyz_max[0], xyz_min[1], xyz_max[2]],  # (x2, y1, z2)
        [xyz_max[0], xyz_max[1], xyz_min[2]],  # (x2, y2, z1)
        [xyz_max[0], xyz_max[1], xyz_max[2]],  # (x2, y2, z2)
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
        [vertices[7], vertices[6], vertices[2], vertices[3]],  # Top
        [vertices[0], vertices[4], vertices[6], vertices[2]],  # Front
        [vertices[1], vertices[5], vertices[7], vertices[3]],  # Back
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # Right
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # Left
    ]
    
    # Plot the faces
    ax.add_collection3d(Poly3DCollection(faces, facecolors='orange', linewidths=1, edgecolors='black', alpha=.1))


if __name__ == "__main__":
    # FILE_NAME = 'Cet_2_02.csv'
    # data_path = f"./data/{FILE_NAME}"
    # rot, trans = read_vicon_file(data_path)
    
    dir_list = load_all_npy('./data/05_01_2024_11_11_06')
    
    file_select = np.random.randint(0, len(dir_list))
    print(f"Selected file: {file_select}")
    
    # 15688
    throw_file = dir_list[15688]
    
    trans = np.load(throw_file) * 1000
    
    coord_start = int(0.8 * len(trans))
    start = trans[coord_start]
    trans = trans - start 
    
    
    xyz_min = np.array([-100, -100, -20])
    xyz_max = np.array([100, 100, 20])
    estimator = Estimator(dt = 1/80, max_time = 0.75, xyz_min = xyz_min , xyz_max = xyz_max)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])
    plot_bbox(ax, xyz_min, xyz_max)
    

    for i, pos in enumerate(trans):
        
            if i < 30:
                
                trajectory, in_workspace = estimator.position_callback(pos)
               
                if trajectory is not None:
                    # xs = trajectory[:estimator.index_smooth_time, 0]
                    # ys = trajectory[:estimator.index_smooth_time, 1]
                    # zs = trajectory[:estimator.index_smooth_time, 2]
                        
                    # ax.scatter(xs,ys, zs, color='#cc8b12')
                    
                    xs = estimator.prev_trajectory[in_workspace][-1, 0]
                    ys = estimator.prev_trajectory[in_workspace][-1, 1]
                    zs = estimator.prev_trajectory[in_workspace][-1, 2]
                    ax.scatter(xs,ys, zs, color='#d11dcb')
                    
                
            if i == 10:        
                if estimator.prev_trajectory is not None:
                    trajectory = estimator.prev_trajectory
                    
                    xs = trajectory[:, 0]
                    ys = trajectory[:, 1]
                    zs = trajectory[:, 2]
                    
                    ax.scatter(xs,ys, zs)
            
            
            
        
    
    velocities = trans[1:] - trans[:-1]
    xs = trans[:, 0]
    ys = trans[:, 1]
    zs = trans[:, 2]
    for xyz, uvw in zip(trans[:-1], velocities):
        x, y, z = xyz
        u, v, w = uvw
        ax.quiver(x, y, z, u, v, w, colors=(1, 0, 0))

    
    # ax.scatter(xs,ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
