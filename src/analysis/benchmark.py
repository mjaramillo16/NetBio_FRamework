import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import subprocess
from datetime import datetime
import math

# --- METRICS & SKLEARN IMPORTS ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 1. PATH SETUP
# ==========================================
_current_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = _current_dir
while _project_root != os.path.dirname(_project_root):
    if os.path.isdir(os.path.join(_project_root, 'src')):
        break
    _project_root = os.path.dirname(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ==========================================
# 2. MODULAR MODEL IMPORTS
# ==========================================
try:
    from src.modeling.neural_ode import HybridNeuralODE
    from src.modeling.Glasso_NeuralODE import Glasso_NeuralODE
    from src.modeling.bayesian_network import LinearGaussianBN
    try:
        from src.modeling.baselines import StandardMLP
    except ImportError:
        from src.modeling.Standard_mlp import StandardMLP
except ImportError as e:
    print(f"\n CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# ==============================================================================
# 3. HELPER FUNCTIONS: FAST TRAINING WRAPPERS
# ==============================================================================
def train_baseline_mlp(model, t_norm_train, target_data, epochs=300, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    target_data = target_data.squeeze()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(t_norm_train)
        loss = loss_fn(pred, target_data)
        loss.backward()
        optimizer.step()
    return model

def train_ode_model(model, x_init, t_span, target_data, epochs=150, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_init, t_span)
        loss = loss_fn(pred, target_data)
        loss.backward()
        optimizer.step()
    return model

# ==============================================================================
# 4. HELPER: STRUCTURAL METRICS & CROSS-VALIDATION
# ==============================================================================
def calculate_structural_metrics(model, model_name, true_adj_tensor):
    metrics = {"L1_Norm": np.nan, "Active_Edges_Pct": np.nan}
    threshold = 1e-4
    try:
        if "Hybrid" in model_name and hasattr(model, 'ode_func') and hasattr(model.ode_func, 'graph_layer'):
            weights = model.ode_func.graph_layer.weights.detach().cpu()
            mask = model.ode_func.graph_layer.adj_mask.detach().cpu()
            masked_weights = weights * mask
            metrics["L1_Norm"] = torch.norm(masked_weights, p=1).item()
            metrics["Active_Edges_Pct"] = (torch.sum(torch.abs(masked_weights) > threshold).item() / torch.sum(mask).item() * 100) if torch.sum(mask).item() > 0 else 0.0
        elif "Glasso" in model_name and hasattr(model, 'ode_func'):
            target_layer = getattr(model.ode_func, 'layer1', None)
            if target_layer is not None and hasattr(target_layer, 'mask'):
                mask = target_layer.mask.detach().cpu()
                metrics["L1_Norm"] = torch.norm(target_layer.weight.detach().cpu() * mask, p=1).item()
                metrics["Active_Edges_Pct"] = (torch.sum(mask).item() / mask.numel() * 100)
        elif "Pure" in model_name and hasattr(model, 'ode_func'):
            layer = getattr(model.ode_func, 'graph_layer', getattr(model.ode_func, 'layer1', None))
            if layer:
                w = layer.weights.detach().cpu() if hasattr(layer, 'weights') else layer.weight.detach().cpu()
                metrics["L1_Norm"] = torch.norm(w, p=1).item()
                metrics["Active_Edges_Pct"] = 100.0
        elif "MLP" in model_name:
            metrics["L1_Norm"] = torch.norm(model.net[0].weight.detach().cpu(), p=1).item()
            metrics["Active_Edges_Pct"] = 0.0
    except Exception: pass
    return metrics

def calculate_rolling_cv_mse(model, model_name, x_tensor, t_span, splits=4):
    """
    Time-Series Cross-Validation: Evaluates the model on rolling forward windows.
    Returns the average MSE across all validation windows.
    """
    if model is None or not hasattr(model, 'eval'):
        return np.nan
        
    n_time = len(t_span)
    if n_time < splits * 2: 
        return np.nan # Not enough data for CV
        
    window_size = n_time // splits
    cv_errors = []
    
    model.eval()
    try:
        for i in range(splits - 1):
            # Take a starting point inside the time series
            start_idx = i * window_size
            end_idx = min(start_idx + window_size * 2, n_time) # Predict one window ahead
            
            x_window = x_tensor[start_idx:end_idx]
            t_window = t_span[start_idx:end_idx]
            
            if "ODE" in model_name:
                with torch.no_grad():
                    pred = model(x_window[0:1], t_window).squeeze(0).cpu().numpy()
            elif "MLP" in model_name:
                t_norm_window = (t_window - t_span.min()) / (t_span.max() - t_span.min())
                with torch.no_grad():
                    pred = model(t_norm_window).cpu().numpy()
            else:
                continue
                
            true_vals = x_window.cpu().numpy()
            mse = mean_squared_error(true_vals, pred)
            cv_errors.append(mse)
            
        return np.mean(cv_errors) if cv_errors else np.nan
    except Exception:
        return np.nan

# ==============================================================================
# 5. CORE BENCHMARK ENGINE 
# ==============================================================================
def run_benchmark_comparison(pkl_file, model_path, device='cpu'):
    with open(pkl_file, 'rb') as f: data = pickle.load(f)
    adj = torch.FloatTensor(data['adj']).to(device)
    x = torch.FloatTensor(data['x']).to(device)
    if x.dim() == 3: x = x[0]

    # ======== NORMALIZATION ========
    x_np = x.cpu().numpy()
    x_np = np.log2(x_np - x_np.min() + 1.0) 
    scaler = MinMaxScaler()
    x_np = scaler.fit_transform(x_np)
    x = torch.FloatTensor(x_np).to(device)
    
    n_nodes, n_time = x.shape[1], x.shape[0]
    t_span = torch.linspace(0, 1, n_time).to(device)
    t_norm = (t_span - t_span.min()) / (t_span.max() - t_span.min()) if t_span.max() > t_span.min() else t_span
    
    split_idx = int(n_time * 0.70)
    has_forecasting = split_idx > 2 and split_idx < n_time
    t_train, t_test = t_span[:split_idx], t_span[split_idx:]
    t_norm_train, t_norm_test = t_norm[:split_idx], t_norm[split_idx:]
    x_train, x_test = x[:split_idx], x[split_idx:]

    predictions, forecast_errors, models_obj = {}, {}, {}
    
    # 1. Linear Regression
    try:
        reg = LinearRegression().fit(t_span.cpu().numpy().reshape(-1, 1), x.cpu().numpy())
        predictions['Linear Regression'] = reg.predict(t_span.cpu().numpy().reshape(-1, 1))
        if has_forecasting:
            pred_f = LinearRegression().fit(t_train.cpu().numpy().reshape(-1, 1), x_train.cpu().numpy()).predict(t_test.cpu().numpy().reshape(-1, 1))
            forecast_errors['Linear Regression'] = mean_squared_error(x_test.cpu().numpy(), pred_f)
    except: pass

    # 2. BN
    try:
        bn = LinearGaussianBN(n_nodes, adj)
        bn.fit(x)
        predictions['Bayesian Network'] = bn.predict_sequence(x[0], n_time)
        if has_forecasting:
            bn_f = LinearGaussianBN(n_nodes, adj)
            bn_f.fit(x_train)
            forecast_errors['Bayesian Network'] = mean_squared_error(x_test.cpu().numpy(), bn_f.predict_sequence(x_train[-1], len(t_test)))
    except: pass

    # 3. MLP
    try:
        mlp = train_baseline_mlp(StandardMLP(n_nodes).to(device), t_norm, x, epochs=300)
        with torch.no_grad(): predictions['Standard MLP'] = mlp(t_norm).cpu().numpy(); models_obj['Standard MLP'] = mlp
        if has_forecasting:
            mlp_f = train_baseline_mlp(StandardMLP(n_nodes).to(device), t_norm_train, x_train, epochs=300)
            with torch.no_grad(): forecast_errors['Standard MLP'] = mean_squared_error(x_test.cpu().numpy(), mlp_f(t_norm_test).cpu().numpy())
    except: pass

    # 4. Pure ODE
    try:
        pure = train_ode_model(HybridNeuralODE(n_nodes, torch.ones((n_nodes, n_nodes))).to(device), x[0:1], t_span, x.unsqueeze(0), epochs=150)
        with torch.no_grad(): predictions['Pure Neural ODE'] = pure(x[0:1], t_span).squeeze(0).cpu().numpy(); models_obj['Pure Neural ODE'] = pure
        if has_forecasting:
            pure_f = train_ode_model(HybridNeuralODE(n_nodes, torch.ones((n_nodes, n_nodes))).to(device), x_train[0:1], t_train, x_train.unsqueeze(0), epochs=150)
            with torch.no_grad(): forecast_errors['Pure Neural ODE'] = mean_squared_error(x_test.cpu().numpy(), pure_f(x_train[0:1], t_span).squeeze(0)[split_idx:].cpu().numpy())
    except: pass

    # 5. Glasso ODE
    try:
        print("   > Training Glasso Neural ODE...")
        glasso_model = Glasso_NeuralODE(n_nodes, input_data=x.cpu().numpy(), alpha=0.1, device=device).to(device)
        glasso = train_ode_model(glasso_model, x[0:1], t_span, x.unsqueeze(0), epochs=150)
        
        with torch.no_grad(): 
            predictions['Glasso Neural ODE'] = glasso(x[0:1], t_span).squeeze(0).cpu().numpy()
            models_obj['Glasso Neural ODE'] = glasso
            
        if has_forecasting:
            glasso_f_model = Glasso_NeuralODE(n_nodes, input_data=x_train.cpu().numpy(), alpha=0.1, device=device).to(device)
            glasso_f = train_ode_model(glasso_f_model, x_train[0:1], t_train, x_train.unsqueeze(0), epochs=150)
            with torch.no_grad(): 
                forecast_errors['Glasso Neural ODE'] = mean_squared_error(
                    x_test.cpu().numpy(), 
                    glasso_f(x_train[0:1], t_span).squeeze(0)[split_idx:].cpu().numpy()
                )
    except Exception as e: 
        print(f"  Glasso Neural ODE failed: {e}")

    # 6. Hybrid ODE
    try:
        model = HybridNeuralODE(n_nodes, adj).to(device)
        if os.path.exists(model_path): model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
        else: model = train_ode_model(model, x[0:1], t_span, x.unsqueeze(0), epochs=150)
        with torch.no_grad(): predictions['Hybrid Neural ODE'] = model(x[0:1], t_span).squeeze(0).cpu().numpy(); models_obj['Hybrid Neural ODE'] = model
        if has_forecasting:
            hybrid_f = train_ode_model(HybridNeuralODE(n_nodes, adj).to(device), x_train[0:1], t_train, x_train.unsqueeze(0), epochs=150)
            with torch.no_grad(): forecast_errors['Hybrid Neural ODE'] = mean_squared_error(x_test.cpu().numpy(), hybrid_f(x_train[0:1], t_span).squeeze(0)[split_idx:].cpu().numpy())
    except: pass

    # Consolidate Metrics
    metrics_list = []
    real_flat = x.cpu().numpy().flatten()
    for name, pred in predictions.items():
        pred_flat = pred.flatten()
        if pred_flat.shape != real_flat.shape: continue
        
        struct = calculate_structural_metrics(models_obj.get(name), name, adj)
        rolling_cv = calculate_rolling_cv_mse(models_obj.get(name), name, x, t_span) # NUEVA MÉTRICA
        
        metrics_list.append({
            "model": name, 
            "MSE_Interpolation": mean_squared_error(real_flat, pred_flat), 
            "Forecast_MSE": forecast_errors.get(name, np.nan), 
            "Rolling_CV_MSE": rolling_cv, # <-- AÑADIDO AL CSV FINAL
            "MAE": mean_absolute_error(real_flat, pred_flat), 
            "R2": r2_score(real_flat, pred_flat), 
            "pearson": pearsonr(real_flat, pred_flat)[0],
            "L1_Norm": struct["L1_Norm"], 
            "Active_Edges_Pct": struct["Active_Edges_Pct"],
            "filename": os.path.basename(pkl_file)
        })
    return pd.DataFrame(metrics_list), predictions, x.cpu().numpy(), t_span.cpu().numpy(), data.get('nodes', [str(i) for i in range(n_nodes)])

# ==============================================================================
# 6. PLOTTING FUNCTIONS 
# ==============================================================================

def plot_scatter_results(real_data, predictions, output_dir, timestamp, case_name):
    real_flat = real_data.flatten()
    n_models = len(predictions)
    cols = min(n_models,2) 
    rows = math.ceil(n_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f'Global Expression Correlation: {case_name}', fontsize=16, y=0.98, fontweight='bold')
    axes_flat = axes.flatten() if n_models > 1 else [axes]
    
    for i, (name, pred) in enumerate(predictions.items()):
        ax = axes_flat[i]
        pred_flat = pred.flatten()
        r2 = r2_score(real_flat, pred_flat)
        
        sns.scatterplot(x=real_flat, y=pred_flat, ax=ax, alpha=0.4, s=25, color="#2b5c8a", edgecolor="w", linewidth=0.2)
        min_v, max_v = min(real_flat.min(), pred_flat.min()), max(real_flat.max(), pred_flat.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, alpha=0.8)
        
        ax.set_title(f"{name}\n(R² = {r2:.3f})", fontweight='bold')
        ax.set_xlabel("True Expression (Normalized)")
        ax.set_ylabel("Predicted Expression")
        ax.grid(True, alpha=0.3, ls='--')
        
    for j in range(i + 1, len(axes_flat)): fig.delaxes(axes_flat[j])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/Scatter_{case_name}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectory_comparison(t_span, real_data, predictions, output_dir, timestamp, case_name, node_names):
    variances = np.var(real_data, axis=0)
    top_genes = np.argsort(variances)[-3:] 
    
    fig, axes = plt.subplots(len(top_genes), 1, figsize=(10, 4 * len(top_genes)), sharex=True)
    fig.suptitle(f'Temporal Dynamics of Key Genes: {case_name}', fontsize=16, y=0.98, fontweight='bold')
    
    colors = sns.color_palette("Set1", n_colors=len(predictions))
    split_time = t_span[int(len(t_span) * 0.70)]
    
    for i, idx in enumerate(top_genes):
        ax = axes[i]
        gene_name = node_names[idx] if idx < len(node_names) else f"Gene {idx}"
        
        ax.plot(t_span, real_data[:, idx], marker='o', color='gray', linestyle='-', 
                linewidth=2, markersize=6, alpha=0.7, label='Ground Truth', zorder=10)
        
        ax.axvline(x=split_time, color='red', linestyle='--', alpha=0.4, label='Forecasting Start')
        
        for j, (name, pred) in enumerate(predictions.items()):
            if "Hybrid" in name:
                ax.plot(t_span, pred[:, idx], linestyle='-', linewidth=3, color='#e74c3c', label=name, zorder=5)
            else:
                ax.plot(t_span, pred[:, idx], linestyle='--', linewidth=1.5, color=colors[j], label=name, alpha=0.8)
                
        ax.set_title(f"Gene: {gene_name} (High Variance)", fontweight='bold')
        ax.set_ylabel("Expression Level")
        ax.grid(True, alpha=0.3, ls='--')
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
            
    axes[-1].set_xlabel("Samples (Time)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/Trajectory_{case_name}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_master_plots(master_df, output_dir, timestamp):
    if master_df.empty: return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    metrics = [
        ("pearson", "Pearson Correlation", False), 
        ("MSE_Interpolation", "MSE (Global Fit)", True), 
        ("Forecast_MSE", "Future Prediction MSE (Forecasting)", True), 
        ("Rolling_CV_MSE", "Time-Series CV MSE (Rolling Horizon)", True), # <-- NUEVO GRÁFICO GENERADO AUTOMÁTICAMENTE
        ("L1_Norm", "Network Sparsity (L1 Norm)", True)
    ]
    for m, t, log in metrics:
        if m in master_df.columns and not master_df[m].isna().all():
            plt.figure(figsize=(10, 6))
            sns.barplot(data=master_df, x="filename", y=m, hue="model", palette="viridis", edgecolor='black')
            if log: plt.yscale("log")
            plt.title(t, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/Bench_{m}_{timestamp}.png", dpi=300)
            plt.close()

def call_visualize_graph(pkl, out, ts):
    script = os.path.join(_project_root, "src/analysis/visualize_graph.py")
    if os.path.exists(script): subprocess.run([sys.executable, script, "--input", pkl, "--output", out, "--timestamp", ts])

# ==============================================================================
# 7. ORCHESTRATOR
# ==============================================================================
def run_full_benchmark_suite(cases_list, output_dir, device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = f"{base_output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Benchmark Suite (Timestamp: {timestamp})")
    print(f" Results will be saved in: {output_dir}")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    all_results = []
    
    for case in cases_list:
        if os.path.exists(case['pkl']):
            df, preds, x_real, t_span, nodes = run_benchmark_comparison(case['pkl'], case['model'], device)
            if not df.empty:
                all_results.append(df)
                cname = os.path.splitext(os.path.basename(case['pkl']))[0]
                plot_scatter_results(x_real, preds, output_dir, timestamp, cname)
                plot_trajectory_comparison(t_span, x_real, preds, output_dir, timestamp, cname, nodes)
                #call_visualize_graph(case['pkl'], output_dir, timestamp)
        else:
            print(f"  SKIPPED: File '{os.path.basename(case['pkl'])}' does not exist in the data folder.")
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(output_dir, f"master_benchmark_{timestamp}.csv"), index=False)
        generate_master_plots(final_df, output_dir, timestamp)
        print(f"\n Benchmark Completed Successfully!")

if __name__ == "__main__":
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join(_project_root, "results/thesis_final_results")
    cases = [{"pkl": os.path.join(_project_root, "data/networks/hsa_hsa04115_graph.pkl"), "model": ""}]
    run_full_benchmark_suite(cases, out, dev)