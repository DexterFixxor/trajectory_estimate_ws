import time
import torch
import torch.nn as nn
from typing import List, Union


class LSTMExtractor(nn.Module):

    def forward(self, x):
        tensor, _ = x
        return tensor


class TrajectoryEstimator(nn.Module):

    def __init__(self, input_size: int, output_size: int, num_lstm: int, hidden_size: int, ff_hidden: list):
        super().__init__()

        lstm_layers = []

        lstm_layers.append(nn.LSTM(input_size, hidden_size, num_layers=num_lstm, batch_first=True))
        # for i in range(num_lstm - 1):
        #     lstm_layers.append(LSTMExtractor())
        #     lstm_layers.append(nn.GRU(hidden_size, hidden_size, batch_first = True))

        lstm_layers.append(LSTMExtractor())
        self.lstm_seq = nn.Sequential(*lstm_layers)

        # feature_extractor = []
        # feature_extractor.append(nn.Flatten()) # [Batch, Seq_len, hidden_size] -> [Batch, Seq_len * hidden_size]

        # num_ff = len(ff_hidden)
        # for i in range(num_ff):
        #     feature_extractor.append(nn.LazyLinear(ff_hidden[i]))
        #     feature_extractor.append(nn.LeakyReLU())
        # feature_extractor.append(nn.LazyLinear(output_size))
        # self.linear = nn.Sequential(*feature_extractor)

        # self.lin1 = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.LeakyReLU()

        self.lin2 = nn.Linear(hidden_size, input_size)

        # initialize weights
        tmp = torch.rand([1, 10, 3])
        self.forward(tmp)

    def forward(self, x):
        x = self.lstm_seq(x)
        # x = self.relu(self.lin1(x))
        x = self.lin2(x)
        return x  # return self.linear(x)

    @torch.no_grad()
    def estimate_trajectory(self, positions: torch.Tensor, desired_len: int, fps: int):
        """Given position vector sequence, estimate trajectory with given len. FPS at which data is
        recorded must be specified, in order to calculate velocities.

        Args:
            positions (torch.Tensor): Sequence of [X,Y,Z] position vectors used to estimate the trajectory.
            desired_len (int): Number of samples to estimate.
        """
        velocities = ((positions[1:] - positions[:-1]))
        trajectory = positions[-1]

        while len(trajectory) <= desired_len:
            est_vel = self.forward(velocities.unsqueeze(0)).squeeze(0)
            # est_delta_positions = est_vel / fps # convert to delta_position vectors
            # est_positions = torch.cumsum(est_delta_positions, 0) + trajectory[-1]   # offset to last position vector
            trajectory = torch.row_stack((trajectory, est_vel))
            velocities = est_vel

        result = torch.cumsum(trajectory, 0)
        return result


# Assuming your TrajectoryEstimator class is already defined above


def benchmark(model, positions, desired_len, fps, warmup=3, runs=100):
    """
    Benchmark wall-clock execution time of model.estimate_trajectory().

    Args:
        model: nn.Module instance
        positions: torch.Tensor input positions
        desired_len: int, trajectory length
        fps: int, frames per second
        warmup: int, number of warmup runs (not timed)
        runs: int, number of timed runs
    """
    # Warmup runs (to stabilize CUDA kernels, JIT, etc.)
    for _ in range(warmup):
        _ = model.estimate_trajectory(positions, desired_len, fps)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.estimate_trajectory(positions, desired_len, fps)

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average wall-clock time over {runs} runs: {avg_time:.6f} seconds")
    print(f"Min: {min(times):.6f}, Max: {max(times):.6f}")
    print(f"Average FPS: {1 / avg_time:.2f}")
    return avg_time, times


if __name__ == "__main__":
    # Example usage
    input_size = 3
    output_size = 3
    num_lstm = 1
    hidden_size = 256
    ff_hidden = []

    model = TrajectoryEstimator(input_size, output_size, num_lstm, hidden_size, ff_hidden)

    # Random input positions [sequence_len, 3]
    positions = torch.rand(10, 3)
    desired_len = 100
    fps = 30

    benchmark(model, positions, desired_len, fps)
