import numpy as np
from .nn_model import TrajectoryEstimator
from .config import all_configs
import torch
import time


class TrajectoryGenerator:

    def __init__(self, dt : float, num_of_points : int, state_dict_path : str, flg_use_nn = True, ):
        """_summary_

        Args:
            dt (float): time between position samples
            num_of_points (int): number of points to generate into the future
            state_dict_path (str): path to the model
            flg_use_nn (bool, optional): Use neural network. If False, analytical model is used. Defaults to True.
        """
        self.dt = dt
        self.num_points = num_of_points
        
        T = np.arange(0, self.dt * self.num_points, self.dt, dtype=np.float64)
        self.times = np.column_stack([T, T, T])
        self.times_squared = self.times ** 2
        self.g = np.array([0.0, 0.0, -9.81]) * 1000 # mm/s^2
        
        self.flg_use_nn = flg_use_nn
        
        cfg = all_configs['best']

        self.model = TrajectoryEstimator(
            input_size=3,
            output_size = cfg['window_size_output'] * 3,
            num_lstm = cfg['num_lstm_layers'],
            hidden_size = cfg['hidden_lstm_neurons'],
            ff_hidden = cfg['ff_hidden']
        )
        
        self.model.load_state_dict(torch.load(state_dict_path, weights_only = True, map_location = torch.device("cpu")))
        self.model.to(torch.device("cpu"))
        self.model.eval()
        
        self.positions = []
        
    def position_callback(self, pos : np.ndarray):
        """Store new position into the list and perform trajectory estimation if min
        number of stored positions match the desired method.

        Args:
            pos (numpy.ndarray): [X,Y,Z] position vector

        Returns:
            (numpy.ndarray | None): Estimated trajectory, if not possible returns None.
        """
        self.positions.append(pos)
        self.positions = self.positions[-11:]
        
        if len(self.positions) <= 10:
            return None
        
        if self.flg_use_nn:
            pos_array = np.array(self.positions, dtype=np.float32)
            
            trajectory = self.model.estimate_trajectory(
                    positions = torch.tensor(pos_array, dtype=torch.float32, requires_grad=False) / 1000.0,  # convert to meters 
                    desired_len = self.num_points, 
                    fps = int(1.0/self.dt)).numpy()
            # convert back to milimeters
            trajectory = trajectory * 1000.0 
        else:
            velocity = (self.positions[-1] - self.positions[-2]) / self.dt
            trajectory = self.positions[-2] + velocity *  self.times + 0.5 * self.g * self.times_squared
        
        return trajectory
        
class ExponentialMovingAverage:
    def __init__(self, alpha : float):
        self.alpha = alpha
        
        self.trajectory = None
        
    def smooth(self, trajectory):
        """
        Args:
            trajectory (numpy.ndarray): Trajectory to be smoothed

        Returns:
            (numpy.ndarray): Smoothed trajectory
        """
        if self.trajectory is not None:
            ema = np.zeros_like(trajectory)
            ema[-1] = trajectory[-1]
            ema[:-1] = self.alpha * trajectory[:-1] + (1 - self.alpha) * self.trajectory[1:]
            self.trajectory = ema
        else:
            self.trajectory = trajectory
            
        return self.trajectory
        
class WorkspaceBoundsCheck:
    
    def __init__(self, xyz_min : list, xyz_max : list):
        """Workspace bound checker, units of min/max values and passed trajectory must be the same.

        Args:
            xyz_min (list): XYZ minimal bounds, in units same as trajectory
            xyz_max (list): XYZ max bounds, in units same as trajectory
        """
        assert (len(xyz_min) == 3 and len(xyz_min) == 3), "Len of XYZ min and max must be equal 3."
        
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        
    def check_bounds(self, trajectory):
        """Check if given trajectory is in bounds. Return will have same len as trajectory, but will be 1D. 

        Args:
            trajectory (_type_): 3D trajectory for bounds check

        Returns:
            (np.ndarray): boolean array weather all emenets (X,Y,Z) for each position individualy fall into bounds
        """
        in_workspace = np.all((trajectory >= self.xyz_min) & (trajectory <= self.xyz_max), axis=1)
        return in_workspace

class MultivariateGaussian:
    def __init__(self, dkl_th : float, keep_last : int = 100, minimum_points : int = 5):    
        """Calculate MultivariateGaussian distribution parameters, and check for DKL between two MVG distributions.

        Args:
            dkl_th (float): treshold to check if DKL has converged
            keep_last (int, optional): Number of last positions to keep after every update_state call. Defaults to 10.
            minimum_points (int, optional): Number of points in positions list, in order to start checking DKL.
        """
        self.positions = []
        
        self.prev_dkl = 10e10
        self.dkl_th = dkl_th
        self.has_converged = False
        
        self.keep_last = keep_last
        self.minimum_points = minimum_points
        
        self.saved_mean = None
        self.saved_cov = None
        
    def update_state(self, pos : np.ndarray):
        
        if len(self.positions) < self.minimum_points:
           self.positions.append(pos)                
        else:
            mean_old = np.mean(self.positions, axis = 0)
            cov_old = np.cov(self.positions, rowvar = False)
            
            self.positions.append(pos)
            
            mean_new = np.mean(self.positions, axis = 0)
            cov_new = np.cov(self.positions, rowvar = False)
            
            dkl = self.dkl(mean1 = mean_old, cov1 = cov_old, mean2 = mean_new, cov2 = cov_new)
            
            self.has_converged = abs(dkl - self.prev_dkl) < self.dkl_th
            self.prev_dkl = dkl

            # Store newest mean and cov for plotting
            self.saved_mean = mean_new
            self.saved_cov = cov_new
            
        self.positions = self.positions[-self.keep_last:]
        
    
    def dkl(self, mean1, cov1, mean2, cov2):
        inv_cov_2 = np.linalg.inv(cov2)
        dkl = 0.5 * (np.log(np.linalg.det(cov2)/np.linalg.det(cov1)) - 3 + 
                     np.trace(inv_cov_2 @ cov1) + 
                     (mean2 - mean1).T @ inv_cov_2 @ (mean2 - mean1))
        
        return dkl
    
class Estimator:
    """Main class for estimating trajectory, check if in workspace, 
    and calculate KL divergence between new and old gaussian distributions of acquired points.
    """
    def __init__(self, 
                 dt : float, 
                 xyz_min : list, 
                 xyz_max : list, 
                 alpha : float = 0.8, 
                 dkl_th : float = 0.05,
                 keep_last_gauss : int = 100, 
                 flg_use_nn : bool = True):
        
        self.dt = dt
        
        self.traj_gen = TrajectoryGenerator(
            dt = self.dt,
            num_of_points = 100,
            state_dict_path = '/home/dexter/Programming/RoboticsFTN/balltrack_ws/src/trajectory_estimator/trajectory_estimator/models/model.pth',
            flg_use_nn = flg_use_nn
        )
        
        self.ema = ExponentialMovingAverage(alpha = alpha)
        self.ws_check = WorkspaceBoundsCheck(xyz_min = xyz_min, xyz_max = xyz_max)
        self.gauss = MultivariateGaussian(dkl_th=dkl_th, keep_last = keep_last_gauss)
        
        self.saved_coord = None
        self.saved_traj = None
        
    def position_callback(self, pos : np.ndarray):
        """Given acquired position vector, generate trajectory if possible. If trajectory is generated, perform smoothing.
        After EMA, check if any of the trajecotry positions fall into WS bounds, if so, use the last one to update the state of 
        Multivariate gaussian distribution. If it has converged, store the latest trajectory.

        Args:
            pos (np.ndarray): _description_
        """
        trajectory = self.traj_gen.position_callback(pos)
        # return trajectory
        if trajectory is not None:
            trajectory = self.ema.smooth(trajectory)
            in_workspace = self.ws_check.check_bounds(trajectory)
            
            if np.any(in_workspace):
                
                last_coord_in_ws = trajectory[in_workspace][-1]
                
                self.gauss.update_state(last_coord_in_ws)
                
                if self.has_converged():
                    self.saved_coord = last_coord_in_ws
                    self.saved_traj = trajectory
    
        
    
    def reset(self):
        self.ema.trajectory = None
        
        self.gauss.positions = []
        self.gauss.saved_mean = None
        self.gauss.saved_cov = None
        self.gauss.prev_dkl = 10e10
        self.gauss.has_converged = False

    def has_converged(self) -> bool:
        return self.gauss.has_converged

    def __str__(self):
        return f"Alpha: {self.ema.alpha}\nDKL treshold: {self.gauss.dkl_th}\nGauss keep last: {self.gauss.keep_last}"
