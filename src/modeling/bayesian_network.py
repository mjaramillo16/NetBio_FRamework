import numpy as np
import torch
from sklearn.linear_model import BayesianRidge

class LinearGaussianBN:
    """
    Rede Bayesiana Dinâmica (DBN) com suposição Linear-Gaussiana.
    Modela a transição de estado x(t) -> x(t+1) respeitando a topologia do grafo.
    """
    def __init__(self, n_nodes, adj_matrix):
        self.n_nodes = n_nodes
        # Garante que a matriz de adjacência seja numpy
        if isinstance(adj_matrix, torch.Tensor):
            self.adj = adj_matrix.cpu().detach().numpy()
        else:
            self.adj = np.array(adj_matrix)
            
        self.models = [None] * n_nodes
        self.parents_map = {}
        
        # Pré-calcula os pais de cada nó (quem regula quem?)
        for i in range(n_nodes):
            self.parents_map[i] = np.where(self.adj[:, i] > 0)[0]

    def fit(self, x_tensor):
        """
        Treina os regressores nó a nó.
        x_tensor: (Time, Nodes) ou (Batch, Time, Nodes)
        """
        # Converte para numpy
        x_np = x_tensor.cpu().detach().numpy()
        
        # Se tiver batch (3D), achata para 2D (Samples, Nodes)
        if x_np.ndim == 3:
            x_np = x_np.reshape(-1, self.n_nodes)
        
        # Prepara pares (t, t+1)
        X_train = x_np[:-1] # Estado atual
        Y_train = x_np[1:]  # Próximo estado
        
        for i in range(self.n_nodes):
            parents = self.parents_map[i]
            target = Y_train[:, i]
            
            # Se não tem pais no grafo, usa auto-regressão (ele mesmo em t-1)
            if len(parents) == 0:
                predictors = X_train[:, i].reshape(-1, 1)
            else:
                predictors = X_train[:, parents]
                
            # Treina Regressão Bayesiana (robusta a poucos dados)
            model = BayesianRidge()
            model.fit(predictors, target)
            self.models[i] = model

    def predict_sequence(self, x0, n_steps):
        """
        Simula uma trajetória futura passo a passo (Rollout).
        x0: Estado inicial (Tensor ou Numpy)
        n_steps: Quantos passos simular
        """
        # Prepara estado inicial
        if isinstance(x0, torch.Tensor):
            current_state = x0.cpu().detach().numpy().flatten()
        else:
            current_state = x0.flatten()
            
        trajectory = [current_state]
        
        # Loop de Simulação (t -> t+1 -> t+2 ...)
        for _ in range(n_steps - 1):
            next_state = np.zeros_like(current_state)
            
            for i in range(self.n_nodes):
                model = self.models[i]
                parents = self.parents_map[i]
                
                # Prepara input baseada nos pais do nó i
                if len(parents) == 0:
                    input_feat = current_state[i].reshape(1, -1)
                else:
                    input_feat = current_state[parents].reshape(1, -1)
                
                # Prediz
                next_state[i] = model.predict(input_feat)[0]
            
            trajectory.append(next_state)
            current_state = next_state # Atualiza para o próximo passo
            
        return np.array(trajectory)