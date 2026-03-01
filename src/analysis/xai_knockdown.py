import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import traceback
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches

# ==========================================
# 1. PATH SETUP (BULLETPROOF)
# ==========================================
_current_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = _current_dir
while _project_root != os.path.dirname(_project_root):
    if os.path.isdir(os.path.join(_project_root, 'src')):
        break
    _project_root = os.path.dirname(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import your base model
try:
    from src.modeling.neural_ode import HybridNeuralODE
except ImportError as e:
    print(f" Error importing HybridNeuralODE in XAI: {e}")
    sys.exit(1)

# ==============================================================================
# 2. AESTHETIC PLOTTING FUNCTIONS (PUBLICATION QUALITY)
# ==============================================================================
def plot_knockdown_impact(t_span, wt_traj, kd_traj, target_gene_name, affected_indices, node_names, output_path, case_name):
    # Estilo limpo de alto rigor científico
    sns.set_theme(style="ticks")
    plt.rcParams.update({'font.family': 'sans-serif'})
    
    n_plots = 1 + len(affected_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots), sharex=True)
    axes = np.atleast_1d(axes)
    
    # Cores Consistentes com os Heatmaps
    color_wt = '#1B4F72'  # Azul Marinho (Normal)
    color_kd = '#E74C3C'  # Vermelho Carmesim (Perturbado)
    color_text = '#2C3E50' # Cinza Escuro (Textos)
    
    fig.suptitle(f'Dynamic Causal Intervention (Virtual Twin)\nIn Silico Knockdown of {target_gene_name} | {case_name}', 
                 fontsize=18, y=0.98 + (0.01/n_plots), fontweight='bold', color=color_text)
    
    # --- 1. Gráfico do Gene Alvo (A Perturbação) ---
    ax_target = axes[0]
    target_idx = list(node_names).index(target_gene_name)
    
    ax_target.plot(t_span, wt_traj[:, target_idx], color=color_wt, ls='-', lw=3, label='Wild-Type Trajectory', zorder=5)
    ax_target.plot(t_span, kd_traj[:, target_idx], color=color_kd, ls='--', lw=3, label=f'{target_gene_name} Silenced', zorder=10)
    
    ax_target.set_title(f"Target Gene: {target_gene_name} (Direct Intervention)", fontweight='bold', fontsize=14, color=color_kd)
    ax_target.set_ylabel("Expression Level", fontweight='bold', color=color_text)
    
    # --- 2. Gráficos dos Genes Afetados a Jusante (Efeito Dominó) ---
    for i, idx in enumerate(affected_indices):
        ax = axes[i + 1]
        gene_name = node_names[idx]
        
        # Preenchimento mostrando o tamanho do impacto causal
        ax.fill_between(t_span, wt_traj[:, idx], kd_traj[:, idx], color=color_kd, alpha=0.15, label='Causal Impact Area')
        
        ax.plot(t_span, wt_traj[:, idx], color=color_wt, ls='-', lw=2.5, alpha=0.85, label='Wild-Type')
        ax.plot(t_span, kd_traj[:, idx], color=color_kd, ls='--', lw=2.5, label='Downstream Response')
        
        ax.set_title(f"Downstream Impact on: {gene_name}", fontweight='bold', fontsize=13, color=color_text)
        ax.set_ylabel("Expression Level", fontweight='bold', color=color_text)

    # --- Estilização Global dos Eixos ---
    for ax in axes:
        ax.grid(True, axis='y', color='#ecf0f1', linestyle='-', linewidth=1)
        ax.grid(False, axis='x') # Remove a grade vertical para focar na linha do tempo
        
        # Legendas sofisticadas
        ax.legend(loc='upper right', frameon=True, edgecolor='#BDC3C7', fontsize=10, shadow=False)
        
        # Define os limites do eixo Y com uma pequena folga para o gráfico respirar
        ax.set_ylim(-0.05, 1.05) 
        sns.despine(ax=ax, trim=True, offset=5)

    axes[-1].set_xlabel("Simulated Time (Continuous Forward Pass)", fontsize=14, fontweight='bold', color=color_text)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ==============================================================================
# 3. PERTURBATION ENGINE (XAI)
# ==============================================================================
def run_xai_analysis(pkl_file, model_path, output_dir, target_gene=None):
    try:
        # 1. Safely Load Data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Safe handling for sparse or dense matrices
        adj_raw = data['adj']
        if hasattr(adj_raw, 'todense'):
            adj = torch.FloatTensor(adj_raw.todense()).to(device)
        else:
            adj = torch.FloatTensor(adj_raw).to(device)
            
        x_raw = data['x']
        if hasattr(x_raw, 'todense'):
            x_raw = x_raw.todense()
        x = torch.FloatTensor(x_raw).to(device)
        
        # Ensure X is 2D (Time, Genes)
        if x.dim() == 3: x = x[0]

        # ======== NORMALIZATION START ========
        x_np = x.cpu().numpy()
        x_np = np.log2(x_np - x_np.min() + 1.0)
        scaler = MinMaxScaler()
        x_np = scaler.fit_transform(x_np)
        x = torch.FloatTensor(x_np).to(device)
        # ======== NORMALIZATION END ========
        
        n_nodes = x.shape[1]
        n_time = x.shape[0]
        t_span = torch.linspace(0, 1, n_time).to(device)
        
        # Ensure node names are strings
        if 'genes' in data: node_names = [str(n) for n in data['genes']]
        elif 'nodes' in data: node_names = [str(n) for n in data['nodes']]
        else: node_names = [f"Gene_{i}" for i in range(n_nodes)]
        
        # 2. Find Target Gene
        if target_gene is None or target_gene not in node_names:
            out_degrees = torch.sum(adj, dim=0).cpu().numpy()
            target_idx = int(np.argmax(out_degrees))
            target_gene = node_names[target_idx]
            print(f"  Automatic Target: {target_gene} (Largest Hub)")
        else:
            target_idx = node_names.index(target_gene)
            print(f"  Manual Target: {target_gene}")

        # 3. Initialize and Load Model
        model = HybridNeuralODE(n_nodes, adj).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # 4. Wild-Type Simulation (Normal Conditions)
        x0_wt = x[0:1].clone() # (1, n_nodes)
        with torch.no_grad():
            out_wt = model(x0_wt, t_span).cpu().numpy()
            wt_trajectory = out_wt.reshape(n_time, n_nodes)

        # 5. Knockdown Simulation
        x0_kd = x[0:1].clone()
        x0_kd[0, target_idx] = 0.0 
        
        with torch.no_grad():
            out_kd = model(x0_kd, t_span).cpu().numpy()
            kd_trajectory = out_kd.reshape(n_time, n_nodes)
            kd_trajectory[:, target_idx] = 0.0 

        # 6. Evaluate Causal Impact
        impact_matrix = np.abs(wt_trajectory - kd_trajectory)
        total_impact_per_gene = np.sum(impact_matrix, axis=0)
        total_impact_per_gene[target_idx] = -1.0 # Ignore the target itself
        
        top_affected_indices = np.argsort(total_impact_per_gene)[-3:][::-1]
        
        # 7. Plot Results
        case_name = os.path.splitext(os.path.basename(pkl_file))[0]
        out_file = os.path.join(output_dir, f"XAI_Knockdown_{target_gene}_{case_name}.png")
        
        plot_knockdown_impact(t_span.cpu().numpy(), wt_trajectory, kd_trajectory, target_gene, 
                              top_affected_indices, node_names, out_file, case_name)
        
        print(f"  Causal Graph saved to: {out_file}")

    except Exception as e:
        print(f" DETAILED XAI ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    pass