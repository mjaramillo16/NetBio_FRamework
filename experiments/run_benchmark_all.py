import torch
import os
import sys
from datetime import datetime

# ==========================================
# 1. PATH SETUP
# ==========================================
_current_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = _current_dir

while _project_root != os.path.dirname(_project_root):
    if os.path.isdir(os.path.join(_project_root, 'src')): break
    _project_root = os.path.dirname(_project_root)

if _project_root not in sys.path: sys.path.insert(0, _project_root)

# ==========================================
# 2. IMPORTS
# ==========================================
try:
    from src.analysis.benchmark import run_full_benchmark_suite
    from src.analysis.xai_knockdown import run_xai_analysis
except ImportError as e:
    print(f"\n IMPORT ERROR: {e}")
    sys.exit(1)

try:
    from src.analysis.benchmark import run_full_benchmark_suite
    from src.analysis.xai_knockdown import run_xai_analysis
    from src.analysis.visualize_all_cases import run_network_visualizations 
except ImportError as e:
    print(f"\n IMPORT ERROR: {e}")
    sys.exit(1)
# ==============================================================================
# 3. CONFIGURATION AND EXECUTION
# ==============================================================================
if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TIMESTAMP CREATION
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # SINGLE OUTPUT FOLDER FOR EVERYTHING (WITH TIMESTAMP)
    OUTPUT_DIR = os.path.join(_project_root, f"results/thesis_results_{timestamp}")
    XAI_DIR = os.path.join(OUTPUT_DIR, "xai_causality")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(XAI_DIR, exist_ok=True)

    # File Definitions (Translated to English)
    thesis_cases = [
        {
            "name": "Cancer (p53)",
            "pkl": os.path.join(_project_root, "data/networks/hsa_hsa04115_graph.pkl"), 
            "model": os.path.join(_project_root, "results/models/case_a_p53.pth")
        },
        {
            "name": "Diabetes (Insulin)",
            "pkl": os.path.join(_project_root, "data/networks/hsa_hsa04910_graph.pkl"), 
            "model": os.path.join(_project_root, "results/models/case_b_insulin.pth")
        },
        {
            "name": "Mouse (Cell Cycle)",
            "pkl": os.path.join(_project_root, "data/networks/mmu_mmu04110_graph.pkl"), 
            "model": os.path.join(_project_root, "results/models/case_c_mouse.pth") 
        },
        {
            "name": "Human (Cell Cycle)",
            "pkl": os.path.join(_project_root, "data/networks/hsa_hsa04110_graph.pkl"), 
            "model": os.path.join(_project_root, "results/models/case_c_human.pth")
        }
    ]

    print(f"Starting Thesis Test Battery on {DEVICE}...")

    # --- FILE VERIFICATION ---
    valid_cases = []
    print("\n Checking data files (.pkl)...")
    for case in thesis_cases:
        if os.path.exists(case['pkl']):
            valid_cases.append(case)
            print(f" Found: {case['name']} -> {os.path.basename(case['pkl'])}")
        else:
            print(f" MISSING: {case['name']}. The file {case['pkl']} does not exist.")
            
    if not valid_cases:
        print("\n NO .pkl files found in data/networks/ folder. Aborting.")
        sys.exit(1)

    # --- EXECUTION 1: BENCHMARK ---
    print("\n--- [STEP 1] Quantitative & Structural Benchmark ---")
    run_full_benchmark_suite(valid_cases, OUTPUT_DIR, DEVICE)

    # --- EXECUTION 2: XAI ---
    print("\n--- [STEP 2] Causal Analysis (XAI - In Silico Knockdown) ---")
    for case in valid_cases:
        if os.path.exists(case['model']):
            try:
                run_xai_analysis(pkl_file=case['pkl'], model_path=case['model'], output_dir=XAI_DIR)
            except Exception as e:
                print(f"  Error in XAI for {case['name']}: {e}")
        else:
            print(f" XAI Skipped for {case['name']}: The trained model (.pth) does not exist.")

    print(f"\n All finished! Check the folder: {OUTPUT_DIR}")

    # ==========================================================
    # --- EXECUTION 3: VISUALIZATIONS: Graph and Heapmap ---
    # ==========================================================
    run_network_visualizations(valid_cases, OUTPUT_DIR)

    print(f"\n All finished! Check the complete thesis results at: {OUTPUT_DIR}")