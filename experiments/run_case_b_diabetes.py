import os
import sys
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.etl import download_file, load_matrix_clean, parse_gpl, harmonize_and_aggregate, save_as_parquet
from src.topology import GraphBuilder, save_graph
from src.modeling import train
from src.analysis.network_auditor import NetworkAuditor  # <--- Nova Auditoria Modular

# --- CONFIGURAÇÃO  ---
CONFIG = {
    "species": "hsa",           
    "pathway": "04910",         # Insulin signaling pathway
    "dataset": "GSE25462",      # Dados de Diabetes
    "gpl": "GPL570",            
    "url_mx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25462/matrix/GSE25462_series_matrix.txt.gz",
    "url_gpl": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
}

PATHS = {
    "raw_mx": f"data/raw/{CONFIG['dataset']}.txt",
    "raw_gpl": f"data/raw/{CONFIG['gpl']}.annot",
    "proc": f"data/processed/{CONFIG['dataset']}_normalized.parquet",
    "net": f"data/networks/{CONFIG['species']}_{CONFIG['species']}{CONFIG['pathway']}_graph.pkl",
    "model": "results/models/case_b_insulin.pth"
}

def main():
    print(f"=== {CONFIG['dataset']} (Diabetes Insulin Resistance) ===")

    # 1. ETL (Extração e Limpeza)
    if not os.path.exists(PATHS['proc']):
        print("--- [FASE 1] ETL ---")
        if not os.path.exists(PATHS['raw_mx']):
             download_file(CONFIG['url_mx'], PATHS['raw_mx'])
        if not os.path.exists(PATHS['raw_gpl']):
             download_file(CONFIG['url_gpl'], PATHS['raw_gpl'])
        
        df = load_matrix_clean(PATHS['raw_mx'])
        mapper = parse_gpl(PATHS['raw_gpl'])
        clean = harmonize_and_aggregate(df, mapper) # Já inclui limpeza para UPPERCASE
        
        if clean is not None:
            save_as_parquet(clean, PATHS['proc'])
        else:
            print("Falha crítica no ETL.")
            return

    # 2. TOPOLOGIA (Com Auditoria Integrada)
    if not os.path.exists(PATHS['net']):
        print("\n--- [FASE 2] Topologia ---")
        gb = GraphBuilder()
        graph = gb.build(CONFIG['species'], CONFIG['pathway'], PATHS['proc'])
        
        if graph: 
            save_graph(graph)
            # Chamada da classe Auditora para validar esparsidade (Gatekeeper)
            is_valid = NetworkAuditor.check(PATHS['net'], "Diabetes Insulin")
            if not is_valid:
                print(" Execução interrompida: Densidade crítica detectada (Quadrado Azul).")
                return
        else:
            print("Falha na construção do grafo.")
            return

    # 3. TREINO (Com Normalização Padronizada para Tese)
    if os.path.exists(PATHS['net']):
        with open(PATHS['net'], 'rb') as f: 
            data = pickle.load(f)
        
        # Auditoria rápida antes de carregar dados de treino
        NetworkAuditor.check(PATHS['net'], "Audit Pre-Training")

        # ======== BLOCO DE NORMALIZAÇÃO PADRONIZADO ========
        x_raw = data['x']
        if hasattr(x_raw, 'todense'): x_raw = x_raw.todense()
        x_np = np.array(x_raw, dtype=np.float32)
        
        # Proteção contra valores negativos
        x_np = np.maximum(x_np, 0) 
        
        # Log2 Condicional: Suaviza picos se os dados forem brutos (> 50)
        if x_np.max() > 50:
            print(f"   > [Normalizer] Valores altos detectados (Max: {x_np.max():.2f}). Aplicando Log2.")
            x_np = np.log2(x_np + 1.0)
        
        # MinMaxScaler (0 a 1) para estabilidade das Neural ODEs
        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(x_np)
        # ===================================================

        print(f"   > Treino iniciado: Matriz {x_norm.shape}")
        train(data['adj'], x_norm, epochs=300, lr=0.005, model_name="case_b_insulin")

if __name__ == "__main__":
    main()