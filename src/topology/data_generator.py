import numpy as np
import pandas as pd
import torch
from scipy.integrate import odeint

class InSilicoBiology:
    """
    Gera dados biológicos sintéticos onde NÓS conhecemos a verdade (Ground Truth).
    Simula uma dinâmica não-linear (Hill Kinetics) típica de regulação gênica.
    """
    def __init__(self, n_genes, sparsity=0.2):
        self.n_genes = n_genes
        np.random.seed(42)
        
        # 1. Cria a Topologia Real (Quem ativa quem?)
        # Matriz esparsa: Maioria é 0, alguns são 1 ou -1 (inibição)
        self.true_adj = np.random.choice([0, 1, -1], size=(n_genes, n_genes), p=[1-sparsity, sparsity/2, sparsity/2])
        np.fill_diagonal(self.true_adj, 0) # Sem auto-loops para simplificar
        
        # Parâmetros cinéticos aleatórios para cada gene
        self.decay = np.random.uniform(0.1, 0.5, n_genes) # Taxa de degradação
        self.beta = np.random.uniform(0.5, 1.0, n_genes)  # Taxa de produção máxima
        self.K = np.random.uniform(0.1, 1.0, n_genes)     # Constante de Michaelis

    def gene_dynamics(self, y, t):
        dydt = np.zeros_like(y)
        for i in range(self.n_genes):
            # Termo de Degradação Natural
            degradation = self.decay[i] * y[i]
            
            # Termo de Síntese (Regulado pelos outros genes)
            synthesis = 0
            regulators = np.where(self.true_adj[:, i] != 0)[0]
            
            if len(regulators) == 0:
                # Se ninguém regula, tem produção basal
                synthesis = self.beta[i] * 0.1
            else:
                # Soma das influências (Hill Function simplificada)
                for reg_idx in regulators:
                    interaction = self.true_adj[reg_idx, i]
                    regulator_conc = y[reg_idx]
                    
                    # Hill function
                    hill = (regulator_conc**2) / (self.K[i]**2 + regulator_conc**2)
                    
                    if interaction == 1: # Ativação
                        synthesis += self.beta[i] * hill
                    elif interaction == -1: # Inibição
                        synthesis += self.beta[i] * (1 - hill)
            
            dydt[i] = synthesis - degradation
        return dydt

    def generate_dataset(self, n_samples, time_points=10, noise_level=0.05):
        """
        Gera o cenário (Poucos Dados ou Big Data).
        """
        t = np.linspace(0, 10, time_points)
        dataset = []
        
        print(f"   > Gerando {n_samples} amostras (trajetórias) de {self.n_genes} genes...")
        
        for _ in range(n_samples):
            # Condição inicial aleatória (cada paciente começa diferente)
            y0 = np.random.uniform(0.1, 2.0, self.n_genes)
            
            # Resolve a EDO real
            trajectory = odeint(self.gene_dynamics, y0, t)
            
            # Adiciona ruído técnico (erro de sequenciamento)
            noise = np.random.normal(0, noise_level, trajectory.shape)
            trajectory_noisy = trajectory + noise
            trajectory_noisy = np.clip(trajectory_noisy, 0, None) # Sem expressão negativa
            
            dataset.append(trajectory_noisy)
            
        # Retorna Tensor no formato (Batch, Time, Genes)
        return torch.FloatTensor(np.array(dataset)), self.true_adj

# ==========================================
# Exemplo de Uso (Simulando os Cenários)
# ==========================================
if __name__ == "__main__":
    # Inicializa o "Mundo Biológico" com 20 genes
    bio_world = InSilicoBiology(n_genes=20, sparsity=0.15)
    
    # CENÁRIO 1: Data Poor (Doenças Raras)
    # Poucos pacientes, alto ruído
    data_small, true_adj = bio_world.generate_dataset(n_samples=20, time_points=10, noise_level=0.2)
    print(f"Cenário 1 (Small): {data_small.shape}") # [20, 10, 20]
    
    # CENÁRIO 2: Big Data (TCGA Comum)
    # Muitos pacientes, ruído moderado
    data_big, _ = bio_world.generate_dataset(n_samples=1000, time_points=10, noise_level=0.1)
    print(f"Cenário 2 (Big): {data_big.shape}")     # [1000, 10, 20]
    
    # Salvar para usar no treino
    torch.save(data_small, "data_small.pt")
    torch.save(data_big, "data_big.pt")
    np.save("ground_truth_adj.npy", true_adj)
    print("Dados salvos.")