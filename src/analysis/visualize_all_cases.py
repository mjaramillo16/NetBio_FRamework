import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings
import pandas as pd
from datetime import datetime
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from networkx.algorithms.community import louvain_communities

# =========================================================
# INTEGRAÇÃO COM ENRICHR (GSEAPY) PARA ANOTAÇÃO BIOLÓGICA
# =========================================================
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False
    print(" [!] 'gseapy' não instalado. A anotação biológica automática será ignorada.")
    print("     Para habilitar, rode: pip install gseapy")

# Configuração de estilo geral (Qualidade de Publicação)
sns.set_theme(style="white")
plt.rcParams.update({'font.family': 'sans-serif'})

def load_graph_data(pkl_path):
    if not os.path.exists(pkl_path):
        return None, None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    adj = data['adj']
    if hasattr(adj, 'todense'):
        adj = np.array(adj.todense())
    else:
        adj = np.array(adj)
        
    nodes = data.get('genes', data.get('nodes', [str(i) for i in range(adj.shape[0])]))
    return adj, nodes

def plot_topology(adj, nodes, out_path, title):
    print(f"  Drawing Topology: {title}")
    G = nx.DiGraph(adj)
    G.remove_nodes_from(list(nx.isolates(G)))
    labels = {i: str(nodes[i]) for i in G.nodes()}
    
    plt.figure(figsize=(16, 16)) 
    
    degrees = dict(G.degree())
    if not degrees:
        plt.close()
        return

    # Escala de tamanho ajustada
    node_sizes = [v * 300 + 600 for v in degrees.values()] 
    
    # Silencia o aviso matemático de componentes desconectados no Kamada-Kawai
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = nx.kamada_kawai_layout(G)
    
    # Cores: Tons frios para nós comuns, tons quentes/vermelhos para os Hubs principais
    nodes_drawn = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=list(degrees.values()), 
        cmap=plt.cm.coolwarm, 
        edgecolors='#2c3e50', 
        linewidths=1.5,
        alpha=0.95
    )
    
    # Arestas sutis e curvadas
    nx.draw_networkx_edges(
        G, pos, 
        width=0.6, 
        alpha=0.3, 
        arrowsize=12, 
        edge_color='#7f8c8d',
        connectionstyle='arc3, rad=0.15'
    )
    
    # Labels com alta legibilidade (Fundo branco quase sólido, texto escuro)
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=10, 
        font_weight='bold',
        font_color='#1a1a1a',
        bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=0.3', alpha=0.85)
    )
    
    plt.title(title, fontsize=22, fontweight='bold', pad=20, color='#2c3e50')
    plt.axis('off')
    
    # Adicionando barra de cores para explicar a importância dos nós (Degree)
    cbar = plt.colorbar(nodes_drawn, fraction=0.03, pad=0.04)
    cbar.set_label('Node Importance (Degree Connectivity)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Export GraphML
    G_export = nx.relabel_nodes(G, labels)
    graphml_path = out_path.replace('.png', '.graphml')
    nx.write_graphml(G_export, graphml_path)

def export_graph_to_csv(adj, nodes, out_path):
    print(f"  Exporting edge list to CSV: {out_path}")
    G = nx.DiGraph(adj)
    labels = {i: str(nodes[i]) for i in G.nodes()}
    G = nx.relabel_nodes(G, labels)
    
    with open(out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Weight'])
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            writer.writerow([u, v, weight])

def reorder_and_find_modules(adj, nodes):
    print("  Detecting communities with Louvain...")
    G = nx.from_numpy_array(adj)
    if G.is_directed():
        G = G.to_undirected()
        
    try:
        communities = list(louvain_communities(G, seed=42)) # Seed para reprodutibilidade
    except Exception as e:
        print(f"  [!] Error detecting communities: {e}. Keeping original order.")
        return adj, nodes, []

    new_order = []
    highlight_modules = []
    start_idx = 0
    colors = ['#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#2980b9', '#d35400', '#16a085', '#c0392b']
    
    for i, comm in enumerate(communities):
        comm_list = list(comm)
        size = len(comm_list)
        new_order.extend(comm_list)
        comm_genes = [nodes[idx] for idx in comm_list] 
        
        highlight_modules.append({
            "start": start_idx,
            "size": size,
            "name": f"Module {i+1}", 
            "color": colors[i % len(colors)],
            "genes": comm_genes,
            "draw_box": size > 2
        })
            
        start_idx += size
        
    adj_reordered = adj[np.ix_(new_order, new_order)]
    nodes_reordered = [nodes[i] for i in new_order]
    
    return adj_reordered, nodes_reordered, highlight_modules

def get_biological_annotation(gene_list, is_mouse=False):
    """
    Usa a API do Enrichr (via gseapy) para encontrar o processo 
    biológico (Gene Ontology) mais significativo para a lista de genes.
    """
    if not HAS_GSEAPY or len(gene_list) < 3:
        return "Unknown (Too few genes)"
    
    organism = 'Mouse' if is_mouse else 'Human'
    gene_sets = 'GO_Biological_Process_2023' 
    
    try:
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=gene_sets,
                         organism=organism,
                         outdir=None) 
        
        if enr.results.empty:
            return "No significant enrichment"
            
        top_result = enr.results.sort_values('Adjusted P-value').iloc[0]
        top_term = top_result['Term'].split(" (GO:")[0]
        pval = top_result['Adjusted P-value']
        
        return f"{top_term.capitalize()} (p={pval:.2e})"
        
    except Exception as e:
        return "Annotation API Error"

def export_communities_to_csv(modules_info, out_path, is_mouse=False):
    """Exporta as comunidades de Louvain e anota automaticamente o Processo Biológico"""
    print(f"  Exporting Louvain communities to CSV: {out_path}")
    
    with open(out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Module_Name', 'Gene_Count', 'Top_Biological_Process', 'Genes_List'])
        
        for mod in modules_info:
            genes_str = ", ".join(mod['genes'])
            
            if mod['size'] >= 3:
                print(f"    Annotating {mod['name']} via Gene Ontology...")
            bio_process = get_biological_annotation(mod['genes'], is_mouse)
            
            writer.writerow([mod['name'], mod['size'], bio_process, genes_str])

def plot_heatmap(adj, nodes, out_path, title, highlight_modules=None):
    print(f"  Drawing Heatmap: {title}")
    
    plt.figure(figsize=(12, 11))
    cmap_custom = ListedColormap(['#F8F9FA', '#1B4F72'])
    
    ax = sns.heatmap(adj, cmap=cmap_custom, cbar=False, 
                     linewidths=0.5, linecolor='white',
                     xticklabels=nodes, yticklabels=nodes)
    
    if highlight_modules:
        for mod in highlight_modules:
            if not mod.get('draw_box', False):
                continue 
                
            start_idx = mod['start']
            size = mod['size']
            color = mod.get('color', '#E74C3C') 
            name = mod['name']
            
            rect = patches.Rectangle(
                (start_idx, start_idx),
                width=size,              
                height=size,             
                linewidth=3.0,           
                edgecolor=color,         
                facecolor='none',        
                linestyle='-',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            plt.text(start_idx + size + 0.8, start_idx + size / 2, 
                     name, color='#2C3E50', va='center', ha='left', 
                     fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.4', alpha=0.9))
    
    font_s = 7 if len(nodes) > 50 else 9
    plt.xticks(rotation=45, ha='right', fontsize=font_s, color='#34495E')
    plt.yticks(rotation=0, fontsize=font_s, color='#34495E')
    
    plt.title(f"Adjacency Matrix (Sparsity)\n{title}", 
              fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    plt.xlabel("Target Genes", fontsize=14, labelpad=15, fontweight='bold', color='#2C3E50')
    plt.ylabel("Source Genes (Regulators)", fontsize=14, labelpad=15, fontweight='bold', color='#2C3E50')
    
    # Legenda corrigida: usando facecolor para evitar conflito
    legend_elements = [
        mpatches.Patch(facecolor='#1B4F72', label='Interaction Present (1)'),
        mpatches.Patch(facecolor='#F8F9FA', label='No Interaction (0)', edgecolor='#BDC3C7', linewidth=1)
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
               title="Edge Connectivity", title_fontproperties={'weight':'bold', 'size':12}, 
               fontsize='11', frameon=True, edgecolor='#BDC3C7', borderpad=1)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def run_network_visualizations(cases_list, output_dir):
    """
    Função principal orquestradora.
    """
    print("\n--- [STEP 3] Generating Network Visualizations (Topology & Heatmaps) ---")
    
    viz_dir = os.path.join(output_dir, "network_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for case in cases_list:
        # Define o title com antecedência para evitar UnboundLocalError
        title = case.get('name', case.get('title', 'Unknown Network'))
        print(f"\n Rendering visualizations for {title}...")
        
        file_path = case.get('pkl', case.get('file'))
        adj, nodes = load_graph_data(file_path)
        
        if adj is not None:
            prefix = os.path.splitext(os.path.basename(file_path))[0]
            
            # 1. Topologia
            plot_topology(adj, nodes, f"{viz_dir}/{prefix}_network.png", f"{title}")
            
            # 2. Heatmap com Módulos
            adj_reordered, nodes_reordered, modules_info = reorder_and_find_modules(adj, nodes)
            plot_heatmap(adj_reordered, nodes_reordered, f"{viz_dir}/{prefix}_heatmap.png", f"{title}", highlight_modules=modules_info)
            
            # 3. Exportações CSV da Rede
            export_graph_to_csv(adj, nodes, f"{viz_dir}/{prefix}_edgelist.csv")
            
            # 4. Anotação Biológica via GSEA
            is_mouse = "Mouse" in title
            export_communities_to_csv(modules_info, f"{viz_dir}/{prefix}_louvain_communities.csv", is_mouse)
            
        else:
            print(f" [!] File not found or invalid: {file_path}")

# Permite rodar o script independentemente
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/Networks_Heatmaps_{timestamp}"
    cases = [
        {"file": "data/networks/hsa_hsa04115_graph.pkl", "title": "p53 Network (Cancer)"},
        {"file": "data/networks/hsa_hsa04910_graph.pkl", "title": "Insulin Network (Diabetes)"},
        {"file": "data/networks/mmu_mmu04110_graph.pkl", "title": "Cellular Cycle (Mouse)"},
        {"file": "data/networks/hsa_hsa04110_graph.pkl", "title": "Cellular Cycle (Human)"}
    ]
    run_network_visualizations(cases, out)
    print(f"\n Visualizations and exports saved successfully in '{out}/'")