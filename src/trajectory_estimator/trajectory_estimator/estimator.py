import numpy as np

class Estimator:
    
    def __init__(self, dt, max_time, xyz_min, xyz_max, alpha):
        self.dt = dt
        self.max_time = max_time
        
        T = np.arange(0, self.max_time, self.dt, dtype=np.float64)
        self.times = np.column_stack([T, T, T])
        self.times_squared = self.times ** 2
        self.g = np.array([0.0, 0.0, -9.81]) * 1000 # mm/s^2

        self.prev_pos = None
        self.velocity = None
        self.prev_trajectory = None
        self.alpha = alpha
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
                    
                    if self.prev_dkl is not None and abs(dkl - self.prev_dkl) < 0.0005:
                        print(dkl, self.prev_dkl)
                        if not self.flg_done:
                            self.saved_trajectory = self.prev_trajectory.copy() # type: ignore
                            self.saved_coord = mean_new.copy()
                            self.flg_done = True
                           
                    self.prev_dkl = dkl
            # ---------- END: check if in workspace
        self.prev_pos = pos
            
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
        
        del self.gaussian_coords
        self.gaussian_coords = []
        self.saved_trajectory = None
        self.saved_coord = None
        
        self.prev_dkl = None
