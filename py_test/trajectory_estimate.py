import csv
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection    
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import transform as t


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
        self.alpha = 0.9
        self.smooth_time = 0.9 * self.max_time
        self.index_smooth_time = int(self.smooth_time / dt)
        
        # Gaussian attributes
        self.gaussian_coords = []
        
        self.saved_trajectory = None
        self.saved_coord = None
        
        self.flg_done = False
        
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
            in_workspace = np.all((self.prev_trajectory >= self.xyz_min) & (self.prev_trajectory <= self.xyz_max), axis=1)
            
            # store last coord on trajectory which is in workspace
            if np.any(in_workspace):
                last_coords = self.prev_trajectory[in_workspace][-1] # type: ignore
                self.gaussian_coords.append(last_coords)
                
                if len(self.gaussian_coords) >= 10:
                    array_cords_old = np.array(self.gaussian_coords[:-1].copy())
                    array_cors_new = np.array(self.gaussian_coords.copy())
                    
                    cov_old = array_cords_old.T @ array_cords_old
                    cov_new = array_cors_new.T @ array_cors_new
                    
                    mean_old = np.mean(array_cords_old, axis=0)
                    mean_new = np.mean(array_cors_new, axis=0)
                    
                    # Calcualte DKL
                    inv_cov_new = np.linalg.inv(cov_new)
                    dkl = 0.5 * (
                        np.log(np.linalg.det(cov_new)/np.linalg.det(cov_old)) - 3 + 
                        np.trace(inv_cov_new @ cov_old) + 
                        (mean_new - mean_old).T @ inv_cov_new @ (mean_new - mean_old)
                        )
                    
                    if self.prev_dkl is not None and abs(dkl - self.prev_dkl) < 0.005 :
                        
                        if self.saved_trajectory is None:
                            self.saved_trajectory = self.prev_trajectory.copy() # type: ignore
                            self.flg_done = True
                            #self.saved_coord = mean_new.copy()
                    
                            self.saved_coord = self.saved_trajectory[in_workspace][-1]
                    
                    self.prev_dkl = dkl
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

    def reset(self):
        self.prev_pos = None
        self.velocity = None
        self.prev_trajectory = None
        
        self.gaussian_coords = [] #self.gaussian_coords[-5:]
        self.saved_trajectory = None
        self.saved_coord = None
        
        self.prev_dkl = None
        
        self.flg_done = False
        
        
        

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
                    "X":  float(row[5]),
                    "Y":  float(row[6]),
                    "Z":  float(row[7])
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
    FILE_NAME = 'D12_proba01.csv'
    data_path = f"./data/{FILE_NAME}"
    rot, trans = read_vicon_file(data_path)
    
    # dir_list = load_all_npy('./data/05_01_2024_11_11_06')   
    # file_select = np.random.randint(0, len(dir_list))
    # print(f"Selected file: {file_select}")
    # # 15688
    # throw_file = dir_list[file_select]
    # trans = np.load(throw_file) * 1000
    
    coord_start = int(0.95 * len(trans))
    start = trans[-10]
    trans = trans - start 
    
    
    xyz_min = np.array([-200.0, -200.0, -200.0])
    xyz_max = np.array([200, 200, 200])
    estimator = Estimator(dt = 1/100, max_time = 1.5, xyz_min = xyz_min , xyz_max = xyz_max)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])
    plot_bbox(ax, xyz_min, xyz_max)
    
    estimator.reset()
    
    start = time.time()
    alpha = 0.1
    for i, pos in enumerate(trans):
            estimator.position_callback(pos)
            if estimator.flg_done:
                trajectory = estimator.saved_trajectory
                print(f"Step: {i}")
                xs = trajectory[:, 0]
                ys = trajectory[:, 1]
                zs = trajectory[:, 2]
                
                ax.scatter(xs,ys, zs)#, color = "#f542ef")
                ax.scatter(
                    estimator.saved_coord[0], 
                    estimator.saved_coord[1], 
                    estimator.saved_coord[2], 
                    s = 100, #size
                    color = [0.0, 1.0, 0.2, alpha]
                    )
                alpha = alpha + 0.1
                alpha = min(1.0, alpha)
                
                estimator.reset()
                
    velocities = trans[1:] - trans[:-1]
    xs = trans[:, 0]
    ys = trans[:, 1]
    zs = trans[:, 2]
    for xyz, uvw in zip(trans[:-1], velocities):
        x, y, z = xyz
        u, v, w = uvw
        ax.quiver(x, y, z, u, v, w, colors=(1, 0, 0))
        
        
        
        
    # trans_new = []
    
    # h_new = np.eye(4)
    # h_new[2, 3] = -46.5
    
    # for i in range(len(rot)):
    #     h_old = np.eye(4)
    #     rot_mat = R.from_euler("XYZ", rot[i], degrees=True).as_matrix()
    #     h_old[:3, :3] = rot_mat.copy()
    #     h_old[:3, 3] = trans[i]
        
    #     h_total = h_old @ h_new
    #     t_new = h_total[:3, 3].copy()
    #     trans_new.append(t_new)
        
    # trans = np.array(trans_new)
    
    # velocities = trans[1:] - trans[:-1]
    # xs = trans[:, 0]
    # ys = trans[:, 1]
    # zs = trans[:, 2]
    # for xyz, uvw in zip(trans[:-1], velocities):
    #     x, y, z = xyz
    #     u, v, w = uvw
    #     ax.quiver(x, y, z, u, v, w, colors=(0, 1, 0))


    # ax.scatter(xs,ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
