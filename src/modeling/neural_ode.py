import torch
import torch.nn as nn
# AQUI ESTAVA O PROBLEMA: Importar do arquivo correto
from .ode_func import ODEFunc 

class HybridNeuralODE(nn.Module):
    """
    O Modelo Principal (Wrapper).
    Gerencia a integração temporal usando RK4.
    """
    def __init__(self, n_nodes, adj_matrix, hidden_dim=64):
        super(HybridNeuralODE, self).__init__()
        self.n_nodes = n_nodes
        
        # Instancia a lógica biológica que está em ode_func.py
        self.ode_func = ODEFunc(n_nodes, adj_matrix, hidden_dim)

    def rk4_step(self, func, t, dt, y):
        k1 = func(t, y)
        k2 = func(t + dt/2, y + dt*k1/2)
        k3 = func(t + dt/2, y + dt*k2/2)
        k4 = func(t + dt, y + dt*k3)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, x_init, t_span):
        device = x_init.device
        batch_size = x_init.size(0)
        n_steps = len(t_span)
        
        trajectory = torch.zeros(batch_size, n_steps, self.n_nodes).to(device)
        trajectory[:, 0, :] = x_init
        
        curr_x = x_init
        
        for i in range(1, n_steps):
            t_prev = t_span[i-1]
            t_curr = t_span[i]
            dt = t_curr - t_prev
            
            curr_x = self.rk4_step(self.ode_func, t_prev, dt, curr_x)
            trajectory[:, i, :] = curr_x
            
        return trajectory