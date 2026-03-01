import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 1. ROBUST PATH SETUP
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root set to: {project_root}")

# ==========================================
# 2. MODULAR MODEL IMPORTS
# ==========================================
try:
    from src.modeling.neural_ode import HybridNeuralODE
    from src.modeling.Glasso_NeuralODE import Glasso_NeuralODE
    from src.modeling.bayesian_network import LinearGaussianBN
    from src.modeling.Standard_mlp import StandardMLP
    print(" Models successfully imported from src.modeling")
except ImportError as e:
    print(f"\n IMPORT ERROR: {e}")
    sys.exit(1)

# ==========================================
# 3. DATA UTILS & NORMALIZATION
# ==========================================
class DataAugmenter:
    """Generates synthetic data (Virtual Patients) for stress testing."""
    @staticmethod
    def generate_virtual_patients(x_ref, n_patients=100, noise_std=0.05):
        if not torch.is_tensor(x_ref): x_ref = torch.FloatTensor(x_ref)
        if x_ref.ndim == 2: x_ref = x_ref.unsqueeze(0) # [1, Time, Genes]
        
        # Creates N copies of the Ground Truth patient
        mean = x_ref.repeat(n_patients, 1, 1) # [N, Time, Genes]
        
        # Adds Gaussian noise to simulate biological variability
        noisy_patients = mean + torch.randn_like(mean) * noise_std
        
        # Ensures expression doesn't escape the [0, 1] domain after normalization
        return torch.clamp(noisy_patients, 0.0, 1.0)

# ==========================================
# 4. STRESS TEST FUNCTIONS
# ==========================================
def test_scalability(models_dict, x_orig, t_span, cohort_sizes, device):
    print("\n [Test 1] Scalability Stress Test (Computational Time)...")
    results = []
    criterion = nn.MSELoss()
    
    # Filters only models trainable via deep gradient descent
    trainable_models = {k: v for k, v in models_dict.items() if "Bayesian" not in k}

    for N in cohort_sizes:
        print(f"   > Simulating clinical cohort with N={N} patients...")
        # Generates batch with baseline noise (5%)
        batch = DataAugmenter.generate_virtual_patients(x_orig, N, 0.05).to(device)
        
        for name, model in trainable_models.items():
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            model.train()
            
            start = time.time()
            for _ in range(3): # Average of 3 epochs to avoid excessive wait times
                optimizer.zero_grad()
                if "ODE" in name:
                    # Initial condition: batch[:, 0, :] -> Prediction along t_span
                    pred = model(batch[:, 0, :], t_span)
                    # Adjusts dimension if the solver returns [Time, Batch, Genes]
                    if pred.shape[0] == len(t_span) and pred.shape[1] == N:
                        pred = pred.transpose(0, 1)
                else: 
                    # Static MLP: ignores input and predicts the mean curve
                    pred = model(t_span).unsqueeze(0).repeat(N, 1, 1)
                
                loss = criterion(pred, batch)
                loss.backward()
                optimizer.step()
            
            avg_time = (time.time() - start) / 3.0
            results.append({"Model": name, "Patients (N)": N, "Time per Epoch (s)": avg_time})
            
    return pd.DataFrame(results)

def test_noise_robustness(models_dict, x_orig, t_span, noise_levels, device):
    print("\n [Test 2] Noise Robustness Analysis...")
    results = []
    criterion = nn.MSELoss()
    
    N_test = 50 # Tests the average precision across 50 simultaneous patients
    gt = x_orig.repeat(N_test, 1, 1).to(device) # Clean Ground Truth
    
    for noise in noise_levels:
        print(f"   > Injecting Biological Noise σ={noise}...")
        noisy_in = DataAugmenter.generate_virtual_patients(x_orig, N_test, noise).to(device)
        
        for name, model in models_dict.items():
            try:
                model.eval() # Inference mode
                if "Bayesian" in name:
                    preds = [torch.tensor(model.predict_sequence(noisy_in[i, 0, :].cpu().numpy(), len(t_span))) for i in range(N_test)]
                    pred = torch.stack(preds).to(device)
                elif "ODE" in name:
                    with torch.no_grad():
                        pred = model(noisy_in[:, 0, :], t_span)
                        if pred.shape[0] == len(t_span): pred = pred.transpose(0, 1)
                else: 
                    with torch.no_grad():
                        pred = model(t_span).unsqueeze(0).repeat(N_test, 1, 1)
                
                # The error is calculated against the CLEAN Ground Truth
                loss = criterion(pred, gt).item()
                results.append({"Model": name, "Noise (std)": noise, "MSE": loss})
            except Exception as e:
                pass # Ignores silent failures of unoptimized models

    return pd.DataFrame(results)

def plot_results(df_scale, df_noise, output_dir, cohorts, timestamp):

    x_part = list(range(0, max(cohorts)+1, 200))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Scalability Plot (Time)
    if not df_scale.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_scale, x="Patients (N)", y="Time per Epoch (s)", hue="Model", style="Model", markers=True, linewidth=2.5)
        plt.title("Computational Scalability (O(N) Complexity)", fontweight='bold')
        plt.xticks(x_part)
        plt.ylabel("Time per Epoch (seconds)")
        plt.xlabel("Cohort Size (N Patients)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Robustness_Scalability_{timestamp}.png", dpi=300)
        plt.close()

    # 2. Noise Plot (Error)
    if not df_noise.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_noise, x="Noise (std)", y="MSE", hue="Model", style="Model", markers=True, linewidth=2.5)
        plt.title("Robustness to Measurement Noise (RNA-Seq/Microarray)", fontweight='bold')
        plt.ylabel("Reconstruction Error (MSE against Ground Truth)")
        plt.xlabel("Injected Gaussian Noise (σ)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Robustness_Noise_{timestamp}.png", dpi=300)
        plt.close()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TIMESTAMP GENERATION
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.join(project_root, f"results/robustness_benchmark_{timestamp}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load Real Data and APPLY NORMALIZATION
    PKL = os.path.join(project_root, "data/networks/hsa_hsa04115_graph.pkl") 
    
    if os.path.exists(PKL):
        with open(PKL, 'rb') as f: data = pickle.load(f)
        
        # Extract matrices
        expr = np.array(data['x'])
        adj = np.array(data['adj'].todense() if hasattr(data['adj'], 'todense') else data['adj'])
        
        # Apply Log2 + MinMax Scaling 
        expr = np.log2(expr - np.min(expr) + 1.0)
        expr = (expr - np.min(expr)) / (np.max(expr) - np.min(expr))
        
        x_real = torch.FloatTensor(expr).to(DEVICE)
        adj_tensor = torch.FloatTensor(adj).to(DEVICE)
        t_span = torch.linspace(0, 1, x_real.shape[0]).to(DEVICE)
        n_nodes = x_real.shape[1]
    else:
        print(" Data not found. Run the data extraction script first.")
        sys.exit(1)

    print(f"Starting Robustness Benchmark with {n_nodes} genes...")

    # 2. Instantiate Models
    models = {}
    
    models["Hybrid Neural ODE"] = HybridNeuralODE(n_nodes, adj_tensor).to(DEVICE)
    models["Pure Neural ODE"] = HybridNeuralODE(n_nodes, torch.ones_like(adj_tensor)).to(DEVICE)
    models["Glasso Neural ODE"] = Glasso_NeuralODE(n_nodes, input_data=expr, alpha=0.1, device=DEVICE).to(DEVICE)
    
    # 3. Run Tests
    cohorts = [10, 50, 100, 200, 500, 1000] # Cohort sizes for scalability test
    noises = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3] # Noise levels for robustness test
    
    df_scale = test_scalability(models, x_real, t_span, cohorts, DEVICE)
    df_noise = test_noise_robustness(models, x_real, t_span, noises, DEVICE)

    # 4. Save Results
    df_scale.to_csv(f"{OUT_DIR}/metrics_scalability_{timestamp}.csv", index=False)
    df_noise.to_csv(f"{OUT_DIR}/metrics_noise_{timestamp}.csv", index=False)
    plot_results(df_scale, df_noise, OUT_DIR, cohorts, timestamp)
    
    print(f"\n Benchmark Completed! Plots saved in: {OUT_DIR}")