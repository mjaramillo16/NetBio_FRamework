import os
import shutil
import subprocess
import sys
from datetime import datetime

# ==========================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLEAN_TARGETS = [
    os.path.join(DATA_DIR, "processed"),
    os.path.join(DATA_DIR, "networks"),
    os.path.join(RESULTS_DIR, "models"),
]

# No seu arquivo reset_and_run.py na raiz
PIPELINE_SCRIPTS = [
    "experiments/run_case_a_cancer.py",   
    "experiments/run_case_b_diabetes.py", 
    "experiments/run_case_c_dual.py",     
    "experiments/run_benchmark_all.py",
    "experiments/robustness_test.py",
    "experiments/Test_Gemeos_real_data.py"
]

def reset_environment():
    """Limpa os diretórios de saída para garantir integridade dos novos dados."""
    print(f"\n=== [{datetime.now().strftime('%H:%M:%S')}] RESETTING ENVIRONMENT ===")
    for target in CLEAN_TARGETS:
        if os.path.exists(target):
            try:
                print(f"  [Cleaning] {os.path.relpath(target, PROJECT_ROOT)}...")
                shutil.rmtree(target)
                os.makedirs(target)
                print(f"  [OK] Cleaned and Recreated.")
            except Exception as e:
                print(f"  [!] Warning cleaning {target}: {e}")
        else:
            os.makedirs(target, exist_ok=True)

def run_pipeline():
    """Executa a sequência de scripts em ordem lógica."""
    print(f"\n=== [{datetime.now().strftime('%H:%M:%S')}] STARTING GLOBAL PIPELINE ===")
    
    for i, script_rel_path in enumerate(PIPELINE_SCRIPTS, 1):
        script_path = os.path.join(PROJECT_ROOT, script_rel_path)
        
        if not os.path.exists(script_path):
            print(f"\n[{i}/{len(PIPELINE_SCRIPTS)}] ERROR: Script not found at: {script_rel_path}")
            
            continue

        print(f"\n[{i}/{len(PIPELINE_SCRIPTS)}]  Executing: {script_rel_path}")
        print("-" * 60)
        
        try:
            # Executa com o cwd na raiz para que as importações 'from src...' funcionem
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                cwd=PROJECT_ROOT 
            )
            process.wait()
            
            if process.returncode != 0:
                print(f"\n[FAIL] {script_rel_path} exited with error code {process.returncode}.")
        except Exception as e:
            print(f"\n[EXCEPTION] Error running {script_rel_path}: {e}")
        print("-" * 60)

if __name__ == "__main__":
    start_time = datetime.now()
    reset_environment()
    run_pipeline()
    
    end_time = datetime.now()
    print(f"\n" + "="*60)
    print(f"GLOBAL THESIS PIPELINE COMPLETED")
    print(f"Total Duration: {end_time - start_time}")
    print("="*60)