import pandas as pd
import networkx as nx
import os
import pickle
import mygene
from bioservices import KEGG

TAXONOMY_MAP = {'hsa': 9606, 'mmu': 10090, 'sce': 4932}

class GraphBuilder:
    def __init__(self):
        self.kegg = KEGG()
        self.mg = mygene.MyGeneInfo()
        
    def _inject_manual_edges(self, G, pathway_id):
        """
        Adiciona arestas críticas que podem faltar no XML do KEGG mas são 
        biologicamente essenciais (Curadoria Manual).
        """
        count = 0
        # Arestas Canônicas para Via da Insulina (hsa04910) e mTOR
        if "04910" in pathway_id or "04150" in pathway_id:
            manual_edges = [
                ("INSR", "IRS1", 1.0),
                ("IGF1R", "IRS1", 1.0),
                ("IRS1", "PIK3CA", 1.0),
                ("PIK3CA", "AKT1", 1.0), # Simplificação PI3K->AKT
                ("AKT1", "MTOR", 1.0),
                ("MTOR", "RPS6KB1", 1.0),
                ("MTOR", "EIF4EBP1", -1.0), # Inibição
                ("AKT1", "GSK3B", -1.0),    # Inibição
                ("PTEN", "AKT1", -1.0)      # Inibição
            ]
            
            print(f"   [TOPOLOGY] Injetando arestas manuais para {pathway_id}...")
            for u, v, w in manual_edges:
                # Só adiciona se ambos os nós existirem no grafo (que veio dos dados)
                if G.has_node(u) and G.has_node(v):
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=w)
                        count += 1
                    else:
                        # Se já existe, força o peso correto (ex: inibição)
                        G[u][v]['weight'] = w
                        
        if count > 0:
            print(f"   > {count} arestas manuais injetadas/corrigidas.")
        return G

    def _parse_kgml_online(self, pid):
        try:
            return self.kegg.parse_kgml_pathway(pid)
        except:
            return None

    def _map_ids_to_symbols(self, entries, species):
        # Coleta todos os IDs brutos
        all_ids = []
        entry_map = {}
        for e in entries:
            # KEGG retorna nomes como "hsa:1234 hsa:5678"
            raw = e.get('name', '').split()
            clean = [x.split(':')[-1] for x in raw]
            all_ids.extend(clean)
            entry_map[e['id']] = clean
            
        tax_id = TAXONOMY_MAP.get(species, 9606)
        try:
            # Consulta em lote no MyGene
            res = self.mg.querymany(all_ids, scopes='entrezgene,kegg', fields='symbol', species=tax_id, verbose=False)
            trans = {}
            for item in res:
                if 'symbol' in item: 
                    trans[item['query']] = item['symbol'].upper()
            return trans, entry_map
        except Exception as e:
            print(f"  Aviso: Falha no MyGene ({e}).")
            return {}, entry_map

    def build(self, species, pathway_id, data_path):
        full_pid = f"{species}{pathway_id.replace(species, '')}"
        
        # 1. Baixar Topologia
        kgml = self._parse_kgml_online(full_pid)
        if not kgml: 
            print("  Falha ao baixar KGML.")
            return None
        
        # 2. Carregar Dados Processados
        if not os.path.exists(data_path):
            print(f" Arquivo de dados não encontrado: {data_path}")
            return None
            
        df = pd.read_parquet(data_path)
        available_genes = set(df.columns)
        
        # 3. Mapear IDs
        sym_map, node_map = self._map_ids_to_symbols(kgml['entries'], species)
        
        G = nx.DiGraph()
        valid_nodes_map = {}
        
        # 4. Adicionar Nós (Interseção KEGG x Dados)
        for entry in kgml['entries']:
            eid = entry['id']
            kids = node_map.get(eid, [])
            found = None
            for k in kids:
                s = sym_map.get(k, k) # Tenta símbolo, senão usa ID
                if s in available_genes:
                    found = s
                    break
            if found:
                G.add_node(found)
                valid_nodes_map[eid] = found
                
        # 5. Adicionar Arestas do KEGG
        for rel in kgml['relations']:
            u, v = rel['entry1'], rel['entry2']
            if u in valid_nodes_map and v in valid_nodes_map:
                # Tenta capturar inibição/ativação do KEGG se possível, ou usa 1.0 padrão
                w = 1.0
                subtype = rel.get('subtypes', [])
                if subtype:
                    name = subtype[0].get('name', '') if isinstance(subtype[0], dict) else str(subtype[0])
                    if 'inhibition' in name: w = -1.0
                
                G.add_edge(valid_nodes_map[u], valid_nodes_map[v], weight=w)
                
        # --- 6. INJEÇÃO MANUAL (Correção da Via) ---
        G = self._inject_manual_edges(G, pathway_id)
        # -------------------------------------------
        
        print(f"   > Grafo Final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
        
        if len(G.nodes) < 5: 
            print("  Grafo muito pequeno. Verifique o mapeamento.")
            return None
        
        nodes = list(G.nodes())
        adj = nx.adjacency_matrix(G, nodelist=nodes).todense()
        x = df[nodes].values
        
        # Transposta para formato (Destino, Origem) -> A_ij = j influencia i
        adj = adj.T 
        
        return {'adj': adj, 'x': x, 'nodes': nodes, 'species': species, 'pathway': full_pid}

def save_graph(graph_data, output_folder="data/networks"):
    """Salva o objeto processado em .pkl"""
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{graph_data['species']}_{graph_data['pathway']}_graph.pkl"
    path = os.path.join(output_folder, filename)
    
    with open(path, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f" Rede Salva em: {path}")

# Teste local rápido
if __name__ == "__main__":
    pass