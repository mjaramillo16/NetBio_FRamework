import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from src.modeling.neural_ode import HybridNeuralODE

def simulate_knockdown(network_path, model_path, target_gene, downstream_genes, output_path):
    """
    Simula o efeito do silenciamento de um gene alvo na rede.
    
    Args:
        target_gene (str): Gene a ser silenciado (ex: 'INSR', 'CDK1').
        downstream_genes (list): Lista de genes alvo para monitorar o efeito.
    """
    print(f"\n   [PERTURBATION] Simulando Knockdown de: {target_gene}")
    
    # 1. Carregar Dados
    if not os.path.exists(network_path):
        print(f"  Erro: Arquivo de rede não encontrado: {network_path}")
        return

    with open(network_path, 'rb') as f:
        data = pickle.load(f)
    nodes = data['nodes']
    adj = torch.FloatTensor(data['adj'])
    x_real = torch.FloatTensor(data['x']) 
    
    if target_gene not in nodes:
        print(f" Gene alvo {target_gene} não encontrado na rede. Pular.")
        return

    idx_target = nodes.index(target_gene)
    
    # 2. Carregar Modelo Treinado
    n_nodes = len(nodes)
    model = HybridNeuralODE(n_nodes=n_nodes, adj_matrix=adj)
    
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"  Erro ao carregar modelo: {e}")
        return

    model.eval()
    
    # 3. Configurar Simulação
    # --- CORREÇÃO DE DIMENSÃO AQUI ---
    # Verifica se x_real é 2D (Time, Genes) ou 3D (Batch, Time, Genes)
    if x_real.dim() == 2:
        # Caso 2D: Pega a primeira linha (t=0) e adiciona dimensão de batch -> (1, Genes)
        x_init = x_real[0:1, :]
    elif x_real.dim() == 3:
        # Caso 3D: Pega a primeira amostra, tempo 0 -> (1, Genes)
        x_init = x_real[0:1, 0, :]
    else:
        print(f" Dimensão de dados inesperada: {x_real.shape}")
        return

    # Define um tempo de simulação padrão (0 a 1)
    t_span = torch.linspace(0, 1, 50)
    
    # --- Simulação WT (Normal) ---
    with torch.no_grad():
        traj_wt = model(x_init, t_span).numpy()
        
    # --- Simulação Knockdown (Mutante) ---
    # Corta as arestas de SAÍDA do gene alvo na matriz de adjacência
    adj_ko = adj.clone()
    adj_ko[idx_target, :] = 0 
    
    # Injeta a matriz "mutada" no modelo
    # (Acessamos o buffer .adj dentro de ode_func)
    model.ode_func.adj = adj_ko 
    
    with torch.no_grad():
        traj_ko = model(x_init, t_span).numpy()
        
    # 4. Plotar Resultados Comparativos
    valid_targets = [g for g in downstream_genes if g in nodes]
    
    # Se não achou os targets específicos, pega os 3 vizinhos mais próximos
    if not valid_targets:
        print(f"  Alvos {downstream_genes} não encontrados. Buscando vizinhos diretos...")
        neighbors_idx = torch.nonzero(adj[idx_target, :], as_tuple=True)[0].tolist()
        valid_targets = [nodes[i] for i in neighbors_idx[:3]]
    
    if not valid_targets:
        print("  Nenhum gene downstream afetado encontrado para plotar.")
        return

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set1", len(valid_targets))
    
    for i, gene in enumerate(valid_targets):
        idx = nodes.index(gene)
        # WT = Linha Sólida
        plt.plot(t_span, traj_wt[0, :, idx], label=f"{gene} (WT)", color=colors[i], ls="-", lw=2, alpha=0.8)
        # KO = Linha Tracejada
        plt.plot(t_span, traj_ko[0, :, idx], label=f"{gene} ({target_gene}-KO)", color=colors[i], ls="--", lw=2)
        
    plt.title(f"Simulação Causal: Impacto do Silenciamento de {target_gene}")
    plt.xlabel("Tempo Virtual (Pseudotempo)")
    plt.ylabel("Expressão Relativa (Normalizada)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"  Gráfico salvo em: {output_path}")