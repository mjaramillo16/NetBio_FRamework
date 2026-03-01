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
from src.analysis.network_auditor import NetworkAuditor

# --- CONFIGURAÇÃO DUAL ---
CONFIGS = [
    {
        "name": "Mouse",
        "species": "mmu",           
        "pathway": "04110",         
        "dataset": "GSE11291",      
        "gpl": "GPL1261",           
        "url_mx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11291/matrix/GSE11291_series_matrix.txt.gz",
        "url_gpl": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz",
    },
    {
        "name": "Human",
        "species": "hsa",           
        "pathway": "04110",         
        "dataset": "GSE10072",      
        "gpl": "GPL96",             
        "url_mx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE10nnn/GSE10072/matrix/GSE10072_series_matrix.txt.gz",
        "url_gpl": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
    }
]

def run_pipeline_for_species(cfg):
    print(f"\n>>> PROCESSANDO: {cfg['name']} ({cfg['species']}) <<<")
    
    path_mx = f"data/raw/{cfg['dataset']}.txt"
    path_gpl = f"data/raw/{cfg['gpl']}.annot"
    path_proc = f"data/processed/{cfg['dataset']}_normalized.parquet"
    path_net = f"data/networks/{cfg['species']}_{cfg['species']}{cfg['pathway']}_graph.pkl"

    # 1. ETL
    if not os.path.exists(path_proc):
        if not os.path.exists(path_mx): download_file(cfg['url_mx'], path_mx)
        if not os.path.exists(path_gpl): download_file(cfg['url_gpl'], path_gpl)
        df = load_matrix_clean(path_mx)
        mapper = parse_gpl(path_gpl)
        clean = harmonize_and_aggregate(df, mapper)
        if clean is not None: save_as_parquet(clean, path_proc)

    # 2. Topologia
    if not os.path.exists(path_net):
        gb = GraphBuilder()
        graph = gb.build(cfg['species'], cfg['pathway'], path_proc)
        if graph: 
            save_graph(graph)
            if not NetworkAuditor.check(path_net, f"Cell Cycle {cfg['name']}"):
                return

    # 3. Treino
    if os.path.exists(path_net):
        with open(path_net, 'rb') as f: data = pickle.load(f)
            
        # ======== BLOCO DE NORMALIZAÇÃO PADRONIZADO ========
        x_raw = data['x']
        if hasattr(x_raw, 'todense'): x_raw = x_raw.todense()
        x_np = np.array(x_raw, dtype=np.float32)
        x_np = np.maximum(x_np, 0) 
        
        if x_np.max() > 50:
            x_np = np.log2(x_np + 1.0)
        
        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(x_np.reshape(-1, x_np.shape[-1])).reshape(x_np.shape) if x_np.ndim == 3 else scaler.fit_transform(x_np)
        # ===================================================

        train(data['adj'], x_norm, epochs=150, lr=0.01, model_name=f"case_c_{cfg['name'].lower()}")

def main():
    for cfg in CONFIGS:
        run_pipeline_for_species(cfg)
    print("\n=== CASO C CONCLUÍDO ===")

if __name__ == "__main__":
    main()