import torch
import torch.nn as nn

class TrajectoryValueNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, traj):  
        B, T, D = traj.shape
        traj = traj.view(B * T, D)
        out = self.net(traj)         
        out = out.view(B, T)
        return out.sum(dim=1)        