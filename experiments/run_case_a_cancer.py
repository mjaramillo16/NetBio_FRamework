import os
import sys
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.etl import download_file, load_matrix_clean, parse_gpl, harmonize_and_aggregate, save_as_parquet
from src.topology import GraphBuilder, save_graph
from src.modeling import train
from src.analysis.network_auditor import NetworkAuditor # <--- Auditoria Modular

# --- CONFIGURAÇÃO ---
CONFIG = {
    "species": "hsa",           
    "pathway": "04115",         # p53 signaling pathway
    "dataset": "GSE25066",      
    "gpl": "GPL96",             
    "url_mx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25066/matrix/GSE25066_series_matrix.txt.gz",
    "url_gpl": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
}

PATHS = {
    "raw_mx": f"data/raw/{CONFIG['dataset']}.txt",
    "raw_gpl": f"data/raw/{CONFIG['gpl']}.annot",
    "proc": f"data/processed/{CONFIG['dataset']}_normalized.parquet",
    "net": f"data/networks/{CONFIG['species']}_{CONFIG['species']}{CONFIG['pathway']}_graph.pkl",
    "model": "results/models/case_a_p53.pth"
}

def main():
    print(f"=== {CONFIG['dataset']} (Cancer p53) ===")

    # 1. ETL
    if not os.path.exists(PATHS['proc']):
        download_file(CONFIG['url_mx'], PATHS['raw_mx'])
        download_file(CONFIG['url_gpl'], PATHS['raw_gpl'])
        df = load_matrix_clean(PATHS['raw_mx'])
        mapper = parse_gpl(PATHS['raw_gpl'])
        clean = harmonize_and_aggregate(df, mapper)
        save_as_parquet(clean, PATHS['proc'])

    # 2. TOPOLOGIA
    if not os.path.exists(PATHS['net']):
        print("\n--- [FASE 2] Construção de Topologia ---")
        gb = GraphBuilder()
        graph = gb.build(CONFIG['species'], CONFIG['pathway'], PATHS['proc'])
        if graph: 
            save_graph(graph)
            # Auditoria de segurança (Gatekeeper)
            if not NetworkAuditor.check(PATHS['net'], "Cancer p53"):
                return

    # 3. TREINO
    if os.path.exists(PATHS['net']):
        with open(PATHS['net'], 'rb') as f: 
            data = pickle.load(f)
        
        # ======== BLOCO DE NORMALIZAÇÃO PADRONIZADO ========
        x_raw = data['x']
        if hasattr(x_raw, 'todense'): x_raw = x_raw.todense()
        x_np = np.array(x_raw, dtype=np.float32)
        x_np = np.maximum(x_np, 0) 

        if x_np.max() > 50:
            print(f"   > [Normalizer] Valores altos (Max: {x_np.max():.2f}). Aplicando Log2.")
            x_np = np.log2(x_np + 1.0)
        
        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(x_np.reshape(-1, x_np.shape[-1])).reshape(x_np.shape) if x_np.ndim == 3 else scaler.fit_transform(x_np)
        # ===================================================

        print(" > Treino iniciado com dados normalizados...")
        train(data['adj'], x_norm, epochs=300, lr=0.005, model_name="case_a_p53")

if __name__ == "__main__":
    main()