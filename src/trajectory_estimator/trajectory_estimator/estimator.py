"""
trajectory_estimator.py
-----------------------
Ball trajectory estimation pipeline.

Pipeline:
    1. TrajectoryGenerator  – takes raw XYZ positions and predicts a future
                              trajectory (either via a neural network or a
                              simple kinematic model).
    2. ExponentialMovingAverage – smooths successive trajectory predictions.
    3. WorkspaceBoundsCheck – filters trajectory points to those inside the
                              robot workspace.
    4. MultivariateGaussian  – tracks whether the landing‑point distribution
                              has converged using KL‑divergence.
    5. Estimator             – glues all four components together.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import all_configs
from .nn_model import TrajectoryEstimator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAVITY_MM_S2 = np.array([0.0, 0.0, -9_810.0])  # mm/s²
MM_TO_M = 1.0 / 1_000.0
M_TO_MM = 1_000.0

WINDOW_SIZE = 11          # total positions stored (10 history + current)
MIN_POSITIONS = WINDOW_SIZE  # minimum needed before predicting


# ---------------------------------------------------------------------------
# TrajectoryGenerator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Predict a fixed‑length future trajectory from a sliding window of
    recent XYZ observations.

    Two prediction modes are supported:
        - Neural network  (``flg_use_nn=True``, default)
        - Kinematic model (``flg_use_nn=False``) – constant velocity + gravity
    """

    def __init__(
        self,
        dt: float,
        num_of_points: int,
        state_dict_path: str,
        flg_use_nn: bool = True,
    ) -> None:
        """
        Args:
            dt:               Time step between position samples (seconds).
            num_of_points:    Number of future positions to predict.
            state_dict_path:  File path to the saved PyTorch model weights.
            flg_use_nn:       Use the neural network when True; kinematic
                              model when False.
        """
        self.dt = dt
        self.num_points = num_of_points
        self.flg_use_nn = flg_use_nn

        # Pre‑compute time vectors for the kinematic fallback -----------------
        # Start at dt (not 0) so the first predicted point is one step *ahead*
        # of the current position, not coincident with it.
        t = np.arange(dt, dt * (num_of_points + 1), dt, dtype=np.float64)

        # Broadcast-friendly shapes: (num_points, 1) so we can multiply by a
        # (3,) velocity/gravity vector without explicit tiling.
        self._t = t[:, np.newaxis]           # (N, 1)
        self._t2 = self._t ** 2               # (N, 1)

        # Sliding window of raw positions (mm), most‑recent last ---------------
        self._positions: list[np.ndarray] = []

        # Load neural network --------------------------------------------------
        self._model = self._load_model(state_dict_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, state_dict_path: str) -> TrajectoryEstimator:
        """Instantiate and load weights into the trajectory network."""
        cfg = all_configs["best"]
        model = TrajectoryEstimator(
            input_size=3,
            output_size=cfg["window_size_output"] * 3,
            num_lstm=cfg["num_lstm_layers"],
            hidden_size=cfg["hidden_lstm_neurons"],
            ff_hidden=cfg["ff_hidden"],
        )
        state = torch.load(state_dict_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model.to("cpu")
        model.eval()
        return model

    def _predict_nn(self) -> np.ndarray:
        """Run the neural network and return a trajectory in mm."""
        pos_tensor = torch.tensor(
            np.array(self._positions, dtype=np.float32) * MM_TO_M,
            requires_grad=False,
        )
        fps = round(1.0 / self.dt)
        trajectory_m = self._model.estimate_trajectory(
            positions=pos_tensor,
            desired_len=self.num_points,
            fps=fps,
        ).numpy()
        return trajectory_m * M_TO_MM

    def _predict_kinematic(self) -> np.ndarray:
        """Constant‑velocity + gravity kinematic prediction (mm)."""
        # velocity estimated from the two most‑recent samples
        velocity = (self._positions[-1] - self._positions[-2]) / self.dt  # (3,)
        origin = self._positions[-1]   # anchor to the *current* position  # (3,)

        # (N, 1) * (3,) broadcasts to (N, 3)
        trajectory = (
            origin
            + velocity * self._t
            + 0.5 * GRAVITY_MM_S2 * self._t2
        )
        return trajectory  # (num_points, 3)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def position_callback(self, pos: np.ndarray) -> np.ndarray | None:
        """Ingest a new XYZ position and, if enough history is available,
        return a predicted trajectory.

        Args:
            pos: 1‑D array ``[X, Y, Z]`` in millimetres.

        Returns:
            Predicted trajectory of shape ``(num_points, 3)`` in millimetres,
            or ``None`` if not enough history has been accumulated yet.
        """
        self._positions.append(pos)

        # Keep only the most recent window to limit memory usage
        self._positions = self._positions[-WINDOW_SIZE:]

        if len(self._positions) < MIN_POSITIONS:
            return None

        return self._predict_nn() if self.flg_use_nn else self._predict_kinematic()


# ---------------------------------------------------------------------------
# ExponentialMovingAverage
# ---------------------------------------------------------------------------

class ExponentialMovingAverage:
    """Smooth successive trajectory predictions with an exponential moving
    average, aligning frames by their *time offset* (not array index).

    For every position except the last, the smoothed value blends the new
    prediction with the *same future instant* from the previous prediction::

        smoothed[i] = alpha * new[i] + (1 - alpha) * prev[i + 1]

    The final point uses the new prediction directly so the trajectory always
    reaches the latest estimate at the horizon.
    """

    def __init__(self, alpha: float) -> None:
        """
        Args:
            alpha: Smoothing factor in ``(0, 1]``.
                   1.0 = no smoothing; smaller = heavier smoothing.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._prev_trajectory: np.ndarray | None = None

    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply EMA and return the smoothed trajectory.

        Args:
            trajectory: Array of shape ``(N, 3)``.

        Returns:
            Smoothed array of the same shape.
        """
        if self._prev_trajectory is None:
            # First call — nothing to blend with, store as‑is.
            self._prev_trajectory = trajectory.copy()
            return self._prev_trajectory

        smoothed = np.empty_like(trajectory)
        # Align by time: new[i] corresponds to the same instant as prev[i+1]
        smoothed[:-1] = (
            self.alpha * trajectory[:-1]
            + (1.0 - self.alpha) * self._prev_trajectory[1:]
        )
        smoothed[-1] = trajectory[-1]

        self._prev_trajectory = smoothed
        return smoothed

    def reset(self) -> None:
        """Discard stored trajectory (e.g. between throws)."""
        self._prev_trajectory = None


# ---------------------------------------------------------------------------
# WorkspaceBoundsCheck
# ---------------------------------------------------------------------------

class WorkspaceBoundsCheck:
    """Test whether trajectory points fall inside an axis‑aligned bounding box.

    Units of ``xyz_min`` / ``xyz_max`` must match the units of trajectories
    passed to :meth:`check_bounds`.
    """

    def __init__(self, xyz_min: list[float], xyz_max: list[float]) -> None:
        """
        Args:
            xyz_min: ``[x_min, y_min, z_min]`` lower corner of the workspace.
            xyz_max: ``[x_max, y_max, z_max]`` upper corner of the workspace.
        """
        # BUG FIX: original code checked len(xyz_min) == 3 twice instead of
        # checking xyz_min AND xyz_max.
        if len(xyz_min) != 3 or len(xyz_max) != 3:
            raise ValueError("xyz_min and xyz_max must each have exactly 3 elements.")
        if np.any(np.array(xyz_min) > np.array(xyz_max)):
            raise ValueError("Each element of xyz_min must be <= the corresponding xyz_max.")

        self.xyz_min = np.array(xyz_min, dtype=np.float64)
        self.xyz_max = np.array(xyz_max, dtype=np.float64)

    def check_bounds(self, trajectory: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating which trajectory points are inside
        the workspace.

        Args:
            trajectory: Array of shape ``(N, 3)``.

        Returns:
            Boolean array of shape ``(N,)``; ``True`` where the point is
            inside the workspace.
        """
        above_min = trajectory >= self.xyz_min  # (N, 3)
        below_max = trajectory <= self.xyz_max  # (N, 3)
        return np.all(above_min & below_max, axis=1)  # (N,)


# ---------------------------------------------------------------------------
# MultivariateGaussian
# ---------------------------------------------------------------------------

class MultivariateGaussian:
    """Maintain a running multivariate Gaussian over observed landing points
    and detect convergence via KL‑divergence.

    Convergence is declared when the *change* in KL‑divergence between
    successive updates falls below ``dkl_th``.
    """

    def __init__(
        self,
        dkl_th: float,
        keep_last: int = 100,
        minimum_points: int = 5,
    ) -> None:
        """
        Args:
            dkl_th:         Convergence threshold on |ΔDKL|.
            keep_last:      Sliding‑window size for stored positions.
            minimum_points: Minimum observations required before testing
                            convergence (must be ≥ 2 for covariance to be
                            well‑defined).
        """
        if minimum_points < 2:
            raise ValueError("minimum_points must be at least 2 to compute covariance.")

        self.dkl_th = dkl_th
        self.keep_last = keep_last
        self.minimum_points = minimum_points

        self._positions: list[np.ndarray] = []
        self._prev_dkl: float = float("inf")

        self.has_converged: bool = False
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(
        mean_p: np.ndarray, cov_p: np.ndarray,
        mean_q: np.ndarray, cov_q: np.ndarray,
    ) -> float:
        """Compute KL(P ‖ Q) for two multivariate Gaussians.

        Formula:
            KL = 0.5 * (log|Σ_q|/|Σ_p| - k + tr(Σ_q⁻¹ Σ_p)
                        + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p))
        where k = 3 (dimensionality).
        """
        k = mean_p.shape[0]
        inv_cov_q = np.linalg.inv(cov_q)
        diff = mean_q - mean_p

        # slogdet avoids det() underflow/overflow with large covariance matrices
        _, logdet_p = np.linalg.slogdet(cov_p)
        _, logdet_q = np.linalg.slogdet(cov_q)
        log_det_ratio = logdet_q - logdet_p
        trace_term = np.trace(inv_cov_q @ cov_p)
        quad_term = diff @ inv_cov_q @ diff  # scalar

        return 0.5 * (log_det_ratio - k + trace_term + quad_term)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_state(self, pos: np.ndarray) -> None:
        """Add a new observation and update convergence status.

        Args:
            pos: 1‑D array ``[X, Y, Z]``.
        """
        if len(self._positions) < self.minimum_points:
            self._positions.append(pos)
            return

        # Snapshot of distribution *before* adding the new point
        pts_old = np.array(self._positions)
        mean_old = pts_old.mean(axis=0)
        cov_old = np.cov(pts_old, rowvar=False)

        self._positions.append(pos)

        # Distribution *after* adding the new point
        pts_new = np.array(self._positions)
        mean_new = pts_new.mean(axis=0)
        cov_new = np.cov(pts_new, rowvar=False)

        dkl = self._kl_divergence(mean_old, cov_old, mean_new, cov_new)

        self.has_converged = abs(dkl - self._prev_dkl) < self.dkl_th
        self._prev_dkl = dkl
        self.mean = mean_new
        self.cov = cov_new

        # Trim to sliding window
        self._positions = self._positions[-self.keep_last:]

    def reset(self) -> None:
        """Clear all state (e.g. between throws)."""
        self._positions = []
        self._prev_dkl = float("inf")
        self.has_converged = False
        self.mean = None
        self.cov = None


# ---------------------------------------------------------------------------
# Estimator  (top‑level orchestrator)
# ---------------------------------------------------------------------------

class Estimator:
    """End‑to‑end pipeline for ball trajectory estimation.

    On each call to :meth:`position_callback`:

    1. Feed the raw position to :class:`TrajectoryGenerator`.
    2. Smooth the resulting trajectory with :class:`ExponentialMovingAverage`.
    3. Find trajectory points inside the robot workspace.
    4. Feed the *last* in‑workspace point to :class:`MultivariateGaussian`.
    5. If the distribution has converged, save the latest trajectory and
       landing coordinate.
    """

    def __init__(
        self,
        dt: float,
        xyz_min: list[float],
        xyz_max: list[float],
        state_dict_path: str,
        alpha: float = 0.8,
        dkl_th: float = 0.05,
        keep_last_gauss: int = 100,
        flg_use_nn: bool = True,
    ) -> None:
        """
        Args:
            dt:               Sampling interval (seconds).
            xyz_min:          Workspace lower bounds ``[x, y, z]`` (mm).
            xyz_max:          Workspace upper bounds ``[x, y, z]`` (mm).
            state_dict_path:  Path to the neural network weights file.
                              (BUG FIX: was hardcoded in original code.)
            alpha:            EMA smoothing factor (0 < alpha ≤ 1).
            dkl_th:           KL‑divergence convergence threshold.
            keep_last_gauss:  Sliding‑window size for the Gaussian estimator.
            flg_use_nn:       Use neural network (True) or kinematic model.
        """
        self.dt = dt

        self.traj_gen = TrajectoryGenerator(
            dt=dt,
            num_of_points=100,
            state_dict_path=state_dict_path,
            flg_use_nn=flg_use_nn,
        )
        self.ema = ExponentialMovingAverage(alpha=alpha)
        self.ws_check = WorkspaceBoundsCheck(xyz_min=xyz_min, xyz_max=xyz_max)
        self.gauss = MultivariateGaussian(dkl_th=dkl_th, keep_last=keep_last_gauss)

        # Outputs set once convergence is reached
        self.saved_coord: np.ndarray | None = None
        self.saved_traj:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def position_callback(self, pos: np.ndarray) -> None:
        """Process a new raw position observation.

        Args:
            pos: 1‑D array ``[X, Y, Z]`` in millimetres.
        """
        trajectory = self.traj_gen.position_callback(pos)
        if trajectory is None:
            return  # Not enough history yet

        trajectory = self.ema.smooth(trajectory)
        in_workspace = self.ws_check.check_bounds(trajectory)

        if not np.any(in_workspace):
            return  # Trajectory never enters the workspace

        # Use the *last* point inside the workspace as the landing estimate
        last_in_ws = trajectory[in_workspace][-1]
        self.gauss.update_state(last_in_ws)

        if self.gauss.has_converged:
            self.saved_coord = last_in_ws
            self.saved_traj = trajectory

    def reset(self) -> None:
        """Reset all stateful components between throws."""
        self.ema.reset()
        self.gauss.reset()
        self.saved_coord = None
        self.saved_traj = None

    @property
    def has_converged(self) -> bool:
        """True once the landing‑point distribution has converged."""
        return self.gauss.has_converged

    def __repr__(self) -> str:
        return (
            f"Estimator("
            f"alpha={self.ema.alpha}, "
            f"dkl_th={self.gauss.dkl_th}, "
            f"keep_last_gauss={self.gauss.keep_last})"
        )
