import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Define a função vetorial f(t, x) que representa dx/dt.
    Antiga GNN_Derivative.
    """
    def __init__(self, n_nodes, adj_matrix, hidden_dim=64):
        super(ODEFunc, self).__init__()
        
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.FloatTensor(adj_matrix)
        
        self.register_buffer('adj', adj_matrix)
        
        self.lin_self = nn.Linear(n_nodes, hidden_dim) 
        self.lin_neighbors = nn.Linear(n_nodes, hidden_dim) 
        self.lin_out = nn.Linear(hidden_dim, n_nodes)
        self.activation = nn.Tanh()

    def forward(self, t, x):
        neighbor_signal = torch.matmul(x, self.adj)
        h = self.lin_self(x) + self.lin_neighbors(neighbor_signal)
        h = self.activation(h)
        dxdt = self.lin_out(h)
        return dxdt