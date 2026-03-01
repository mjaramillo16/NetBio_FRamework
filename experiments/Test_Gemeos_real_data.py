import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Ensures python finds the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling.neural_ode import HybridNeuralODE

# =====================================================================
# MULTI-OMICS WRAPPER: Adds the DNA layer without breaking the original model
# =====================================================================
class MultiOmicsODEFunc(nn.Module):
    def __init__(self, base_ode_func):
        super().__init__()
        self.base_ode_func = base_ode_func
        self.current_mutation = None
        self.mutation_weight = 2.0  # Strength of the mutation's 'Gain of Function'

    def forward(self, t, x):
        # 1. Calculates normal derivative (RNA + KEGG Topology only)
        dx_dt = self.base_ode_func(t, x)
        
        # 2. Injects the continuous effect of Genomic Mutation (DNA), if any
        if self.current_mutation is not None:
            dx_dt = dx_dt + (self.mutation_weight * self.current_mutation)
            
        return dx_dt
# =====================================================================

def run_all_cases_twins():
    # TIMESTAMP GENERATION
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"=== Multi-Omics Virtual Twins: Global Validation (Timestamp: {timestamp}) ===")
    
    output_dir = f"results/Twins_Test_{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration of the 4 scenarios
    configs = [
        {
            "id": "Case A",
            "title": "Case A: p53 Signaling (Human Breast Cancer)",
            "file": "data/networks/hsa_hsa04115_graph.pkl",
            "mut_gene": "TP53",
            "target_gene": "CDKN1A" # p21
        },
        {
            "id": "Case B",
            "title": "Case B: Insulin Cascade (Human Diabetes)",
            "file": "data/networks/hsa_hsa04910_graph.pkl",
            "mut_gene": "AKT3",
            "target_gene": "MTOR" # Metabolic target
        },
        {
            "id": "Case C (Human)",
            "title": "Case C: Cell Cycle (Human NSCLC Tumor)",
            "file": "data/networks/hsa_hsa04110_graph.pkl",
            "mut_gene": "CHEK1",
            "target_gene": "PCNA" # Proliferation marker
        },
        {
            "id": "Case C (Mouse)",
            "title": "Case C: Cell Cycle (Mouse Stem Cells)",
            "file": "data/networks/mmu_mmu04110_graph.pkl",
            "mut_gene": "CHEK1",
            "target_gene": "PCNA" # Proliferation marker
        }
    ]
    
    # Prepare the 2x2 Figure
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, cfg in enumerate(configs):
        print(f"\n Processing {cfg['id']}...")
        
        if not os.path.exists(cfg['file']):
            print(f"   [!] File not found: {cfg['file']}")
            continue
            
        with open(cfg['file'], "rb") as f:
            data = pickle.load(f)
            
        adj = torch.tensor(data['adj'], dtype=torch.float32)
        nodes = list(data['nodes'])
        n_nodes = len(nodes)
        
        # Find Indices with safety Fallback
        mut_gene = cfg['mut_gene']
        target_gene = cfg['target_gene']
        
        nodes_upper = [str(n).upper() for n in nodes]
        
        if mut_gene.upper() in nodes_upper:
            idx_mut = nodes_upper.index(mut_gene.upper())
            actual_mut_gene = nodes[idx_mut]
        else:
            idx_mut = 0
            actual_mut_gene = nodes[0]
            
        if target_gene.upper() in nodes_upper:
            idx_alvo = nodes_upper.index(target_gene.upper())
            actual_target_gene = nodes[idx_alvo]
        else:
            idx_alvo = -1
            actual_target_gene = nodes[-1]
            
        print(f"   > Mutation Injected in: {actual_mut_gene} | Observing target: {actual_target_gene}")
        
        # Initialize Model
        torch.manual_seed(42)
        model = HybridNeuralODE(n_nodes, adj)
        
        # ---> Replace internal function with Multi-omics Wrapper
        model.ode_func = MultiOmicsODEFunc(model.ode_func)
                
        # Configure Twins
        x_init_wt = torch.full((1, n_nodes), 0.5) 
        x_init_mt = torch.full((1, n_nodes), 0.5)
        
        mut_wt = torch.zeros((1, n_nodes)) 
        mut_mt = torch.zeros((1, n_nodes))
        mut_mt[0, idx_mut] = 1.0 # Injects mutation only in the chosen gene
        
        t_span = torch.linspace(0, 5, 50)
        
        # Simulate
        with torch.no_grad():
            # Passes WT mutation (Zero) and simulates time using standard forward(x, t)
            model.ode_func.current_mutation = mut_wt
            traj_wt = model(x_init_wt, t_span)
            
            # Passes MT mutation (One) and simulates time
            model.ode_func.current_mutation = mut_mt
            traj_mt = model(x_init_mt, t_span)
            
        # .squeeze() removes the useless batch dimension (1), leaving only [time, genes]
        y_wt = traj_wt.squeeze()[:, idx_alvo].numpy()
        y_mt = traj_mt.squeeze()[:, idx_alvo].numpy()
        t_np = t_span.numpy()
        
        # Plot on the correct Subplot
        ax = axes[i]
        ax.plot(t_np, y_wt, color='#2980b9', linestyle='-', linewidth=2.5, label='Twin A (WT)')
        ax.plot(t_np, y_mt, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Twin B (Mut: {actual_mut_gene})')
        ax.fill_between(t_np, y_wt, y_mt, color='#e74c3c', alpha=0.15)
        
        ax.set_title(cfg['title'], fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel("Simulated Time (t)", fontsize=11)
        ax.set_ylabel(f"Expression: {actual_target_gene}", fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        
    plt.suptitle("Multi-Omics Validation: Genomic Impact Across 4 Distinct Biological Contexts", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"Real_Multiomics_Twins_All_Cases_{timestamp}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n SUCCESS! Plot saved in: {out_file}")

if __name__ == "__main__":
    run_all_cases_twins()