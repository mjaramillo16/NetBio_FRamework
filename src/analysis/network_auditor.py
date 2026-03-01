import pickle
import numpy as np
import os

class NetworkAuditor:
    """Classe para auditoria de integridade e esparsidade biológica das redes."""
    
    @staticmethod
    def check(pkl_path, label="Network"):
        if not os.path.exists(pkl_path):
            print(f"[{label}] Arquivo não encontrado: {pkl_path}")
            return False

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        adj = data['adj']
        if hasattr(adj, 'todense'):
            adj = adj.todense()
        adj = np.array(adj)
        
        nodes = data.get('nodes', data.get('genes', []))
        n_nodes = adj.shape[0]
        n_edges = np.count_nonzero(adj)
        
        # Cálculo de Densidade
        possible_edges = n_nodes * (n_nodes - 1)
        density = (n_edges / possible_edges) if possible_edges > 0 else 0
        
        print(f"\n--- [AUDITORIA: {label}] ---")
        print(f"  > Densidade: {density:.2%}")
        print(f"  > Genes: {n_nodes} | Arestas: {n_edges}")

        # Gatekeeper: Se a densidade for > 30%, algo está errado (Quadrado Azul)
        if density > 0.30:
            print(f"  ERRO CRÍTICO: Densidade de {density:.2%} detectada!")
            print(f"  Causa provável: Falha na filtragem do KEGG em src/topology.py")
            return False
        
        print(f" Integridade validada.")
        return True