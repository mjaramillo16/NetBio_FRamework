import torch
import torch.nn as nn
import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler 

# ==========================================
# 1. Camada Linear Esparsa
# ==========================================
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('mask', mask)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return nn.functional.linear(input, masked_weight, self.bias)

# ==========================================
# 2. Função Diferencial Informada
# ==========================================
class MaskedODEFunc(nn.Module):
    def __init__(self, n_nodes, adj_matrix, hidden_dim=64):
        super(MaskedODEFunc, self).__init__()
        self.n_nodes = n_nodes
        self.layer1 = MaskedLinear(n_nodes, n_nodes, mask=adj_matrix.float())
        self.layer2 = nn.Sequential(nn.Tanh(), nn.Linear(n_nodes, n_nodes))

    def forward(self, t, y):
        x = self.layer1(y)
        dydt = self.layer2(x)
        return dydt

# ==========================================
# 3. O Modelo Principal (Glasso + Neural ODE)
# ==========================================
class Glasso_NeuralODE(nn.Module):
    def __init__(self, n_nodes, input_data=None, adj_matrix=None, alpha=0.1, device='cpu'):
        super(Glasso_NeuralODE, self).__init__()
        self.n_nodes = n_nodes
        self.device = device

        # --- Passo 1: Inferência de Topologia Robusta ---
        if adj_matrix is None and input_data is not None:
            print(f"   [Glasso] Inferindo topologia (alpha={alpha})...")
            try:
                # 1. Padronização (Crucial para o Glasso não explodir)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(input_data)
                
                # 2. Adição de "Jitter" (Ruído numérico minúsculo para garantir convergência SPD)
                jitter = np.random.normal(0, 1e-5, scaled_data.shape)
                robust_data = scaled_data + jitter

                # 3. Inferência
                glasso = GraphicalLasso(alpha=alpha, max_iter=3000, assume_centered=True)
                glasso.fit(robust_data)
                precision = glasso.precision_
                
                adj_matrix = np.abs(precision) > 1e-5
                np.fill_diagonal(adj_matrix, 0)
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
                print("   [Glasso] Topologia inferida com sucesso!")
                
            except Exception as e:
                print(f"   [Glasso] Falha na inferência ({e}). Usando Matriz Cheia (Pure ODE Fallback).")
                adj_matrix = torch.ones((n_nodes, n_nodes)).to(device)
        
        elif adj_matrix is not None:
            self.adj_matrix = adj_matrix.to(device)
        else:
            adj_matrix = torch.ones((n_nodes, n_nodes)).to(device)

        # --- Passo 2: Construção da ODE ---
        self.ode_func = MaskedODEFunc(n_nodes, adj_matrix).to(device)

    def rk4_step(self, func, t, dt, y):
        k1 = func(t, y)
        k2 = func(t + dt/2, y + dt*k1/2)
        k3 = func(t + dt/2, y + dt*k2/2)
        k4 = func(t + dt, y + dt*k3)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, x_init, t_span):
        t_span = t_span.to(self.device)
        batch_size = x_init.size(0)
        n_steps = len(t_span)
        trajectory = torch.zeros(batch_size, n_steps, self.n_nodes).to(self.device)
        trajectory[:, 0, :] = x_init
        curr_x = x_init
        
        for i in range(1, n_steps):
            t_prev = t_span[i-1]
            t_curr = t_span[i]
            dt = t_curr - t_prev
            curr_x = self.rk4_step(self.ode_func, t_prev, dt, curr_x)
            trajectory[:, i, :] = curr_x
        return trajectory