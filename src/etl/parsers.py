# src/etl/parsers.py

import pandas as pd
import os
import io

def load_matrix_clean(filepath):
    print(f"[PARSER] Lendo matriz: {os.path.basename(filepath)}")
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        start = 0
        for i, line in enumerate(lines):
            if "ID_REF" in line:
                start = i
                break
                
        df = pd.read_csv(io.StringIO("".join(lines[start:])), sep="\t", index_col=0)
        
        # LIMPEZA AGRESSIVA DE IDs
        # Remove aspas, espaços e converte para maiúsculo
        df.index = df.index.astype(str).str.replace('"', '').str.strip().str.upper()
        
        if "!SERIES_MATRIX_TABLE_END" in df.index: 
            df = df.drop("!SERIES_MATRIX_TABLE_END")
            
        return df
    except Exception as e:
        print(f" Erro parser matriz: {e}")
        return None

def parse_gpl(gpl_path, target_col="Gene Symbol"):
    print(f"[PARSER] Processando GPL: {os.path.basename(gpl_path)}")
    if not os.path.exists(gpl_path): return {}

    mapping = {}
    try:
        with open(gpl_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            
        start = 0
        for i, line in enumerate(lines):
            if "!platform_table_begin" in line: 
                start = i + 1
                break
            
        header = lines[start].strip().split('\t')
        
        # Tenta achar a coluna alvo de várias formas
        col_idx = -1
        possible_names = [target_col, "Gene Symbol", "GENE_SYMBOL", "Gene_Symbol", "ORF"]
        
        for name in possible_names:
            for i, h in enumerate(header):
                if name.lower() == h.lower(): 
                    col_idx = i
                    print(f"   > Coluna de gene encontrada: '{h}' (Índice {i})")
                    break
            if col_idx != -1: break
            
        if col_idx == -1:
            print(f" Coluna de gene não encontrada. Headers: {header[:5]}...")
            return {}

        for line in lines[start+1:]:
            if "!platform_table_end" in line: break
            parts = line.strip().split('\t')
            if len(parts) > col_idx:
                probe = parts[0].replace('"', '').strip().upper()
                gene = parts[col_idx].replace('"', '').split('///')[0].strip().upper()
                
                # Ignora genes vazios ou '---'
                if gene and gene != '---' and gene != '' and 'nan' not in gene.lower(): 
                    mapping[probe] = gene
                    
        print(f"   > Mapa com {len(mapping)} sondas.")
        return mapping
    except Exception as e:
        print(f" Erro parser GPL: {e}")
        return {}