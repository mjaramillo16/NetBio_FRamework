import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from .neural_ode import HybridNeuralODE

def train(adj_matrix, data, epochs=500, lr=0.01, device='cpu', model_name='model'):
    """
    Função Genérica de Treinamento.
    
    Args:
        adj_matrix (numpy array ou Tensor): Matriz de Adjacência do Grafo (NxN).
        data (pd.DataFrame ou Tensor): Dados de Expressão (Amostras/Tempo x Genes).
        epochs (int): Número de épocas de treino.
        lr (float): Taxa de aprendizado.
        device (str): 'cpu' ou 'cuda'.
        model_name (str): Nome para salvar o arquivo .pth.
    
    Returns:
        model (nn.Module): Modelo treinado.
        loss_history (list): Histórico de erros.
    """
    print(f"\n[TRAINER] Iniciando treino para: {model_name}")
    
    # 1. Preparação de Dados
    # Se for DataFrame, converte para Tensor
    if isinstance(data, pd.DataFrame):
        # Assume formato (Amostras, Genes)
        x_tensor = torch.FloatTensor(data.values).to(device)
    else:
        x_tensor = torch.FloatTensor(data).to(device)

    # Se a matriz de adjacência for numpy, converte
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.FloatTensor(adj_matrix).to(device)
    else:
        adj_matrix = adj_matrix.to(device)

    # Configuração de Dimensionalidade
    # Precisamos adicionar uma dimensão de Batch se não existir
    # Para séries temporais únicas (Média), Batch=1
    if x_tensor.dim() == 2:
        # (Time, Genes) -> (1, Time, Genes)
        x_tensor = x_tensor.unsqueeze(0)
    
    batch_size, n_steps, n_nodes = x_tensor.shape
    
    print(f"   > Dados: {n_nodes} genes, {n_steps} pontos temporais.")
    
    # Definição do Tempo (Normalizado [0, 1])
    t_span = torch.linspace(0, 1, n_steps).to(device)
    
    # 2. Inicialização do Modelo
    model = HybridNeuralODE(n_nodes=n_nodes, adj_matrix=adj_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Erro Quadrático Médio para regressão temporal
    
    # Estado Inicial (t=0) para todas as amostras
    x_init = x_tensor[:, 0, :] # (Batch, Genes)
    
    # 3. Loop de Treinamento
    loss_history = []
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # A. Forward Pass (Integração da ODE)
        # O modelo recebe o estado inicial e integra até o tempo final
        pred_trajectory = model(x_init, t_span)
        
        # B. Cálculo da Perda
        # Compara a trajetória simulada (pred) com a real (x_tensor)
        loss = criterion(pred_trajectory, x_tensor)
        
        # C. Backward Pass (Backpropagation through Time)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"   > Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    # 4. Salvamento
    os.makedirs("results/models", exist_ok=True)
    save_path = f"results/models/{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f" Modelo salvo em: {save_path}")
    
    return model, loss_history