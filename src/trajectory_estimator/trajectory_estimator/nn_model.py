import torch
import torch.nn as nn
from typing import List, Union


class LSTMExtractor(nn.Module):
    
    def forward(self, x):
        tensor, _ = x
        return tensor
        
class TrajectoryEstimator(nn.Module):

    def __init__(self, input_size : int, output_size : int, num_lstm : int, hidden_size : int, ff_hidden : list):
        super().__init__()


        lstm_layers = []
        
        lstm_layers.append(nn.LSTM(input_size, hidden_size, num_layers=num_lstm, batch_first = True))
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
        return x #return self.linear(x)
    
    @torch.no_grad
    def estimate_trajectory(self, positions : torch.Tensor, desired_len : int, fps : int):
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
            #est_delta_positions = est_vel / fps # convert to delta_position vectors
            #est_positions = torch.cumsum(est_delta_positions, 0) + trajectory[-1]   # offset to last position vector
            trajectory = torch.row_stack((trajectory, est_vel))
            velocities = est_vel

        result = torch.cumsum(trajectory, 0)
        return result





if __name__ == "__main__":
    
        """
        Model test
        """
        batch_size = 16
        seq_len = 20
        in_size = 3
        k = 1
        out_size = k * seq_len * in_size
        
        num_lstm = 1
        
        hidden_size = 256
        ff_hidden = []
        model = TrajectoryEstimator(in_size, out_size, num_lstm, hidden_size, ff_hidden)
        
        x = torch.rand([batch_size, seq_len, in_size])
        y = model(x)
    
        print(model)
        print(y.shape)
        
