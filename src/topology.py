import numpy as np
import pandas as pd
import pickle
import os
from bioservices import KEGG

class GraphBuilder:
    def __init__(self):
        self.kegg = KEGG()

    def build(self, species, pathway_id, parquet_path):
        pathway_name = f"{species}{pathway_id}"
        print(f"[TOPOLOGY] Fetching real biological edges for: {pathway_name}")
        
        # 1. Carregar os genes que você tem no seu dado normalizado
        df = pd.read_parquet(parquet_path)
        dataset_genes = set(df.columns)
        
        # 2. Obter a via do KEGG
        res = self.kegg.get(pathway_name)
        if isinstance(res, int): 
            print(f"  Error: Could not find pathway {pathway_name} in KEGG.")
            return None

        # 3. Parse das relações (Arestas reais)
        kgml = self.kegg.parse_kgml_pathway(pathway_name)
        edges = []
        for rel in kgml['relations']:
            entry1_id = rel['entry1']
            entry2_id = rel['entry2']
            
            # Busca os nomes dos genes para esses IDs internos do KEGG
            gene1_names = self._get_gene_names(kgml, entry1_id)
            gene2_names = self._get_gene_names(kgml, entry2_id)
            
            for g1 in gene1_names:
                for g2 in gene2_names:
                    if g1 in dataset_genes and g2 in dataset_genes:
                        edges.append((g1, g2))

        # 4. Criar a Matriz de Adjacência Esparsa
        gene_list = sorted(list(dataset_genes))
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
        adj = np.zeros((len(gene_list), len(gene_list)))
        
        for g1, g2 in edges:
            adj[gene_to_idx[g1], gene_to_idx[g2]] = 1
            
        print(f"  Success: Generated sparse network with {len(gene_list)} genes and {len(edges)} real edges.")
        
        return {
            'adj': adj,
            'x': df.values,
            'nodes': gene_list,
            'pathway': pathway_name
        }

    def _get_gene_names(self, kgml, entry_id):
        names = []
        for entry in kgml['entries']:
            if entry['id'] == entry_id:
                # O KEGG costuma retornar "hsa:123, hsa:456...", pegamos o Symbol
                raw_names = entry['name'].split()
                for r in raw_names:
                    symbol = self.kegg.get(r)
                    if isinstance(symbol, str) and "SYMBOL" in symbol:
                        # Extrai o símbolo (ex: AKT1) do texto bruto do KEGG
                        for line in symbol.split("\n"):
                            if line.startswith("SYMBOL"):
                                names.append(line.replace("SYMBOL", "").strip().split(",")[0])
        return names

def save_graph(graph_data):
    path = f"data/networks/{graph_data['pathway']}_graph.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"  Graph saved at: {path}")