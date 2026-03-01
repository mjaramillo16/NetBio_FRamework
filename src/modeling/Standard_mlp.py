import torch
import torch.nn as nn

class StandardMLP(nn.Module):
    def __init__(self, n_nodes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_nodes)
        )
    def forward(self, t):
        if t.dim() == 1: t = t.unsqueeze(1)
        return self.net(t)