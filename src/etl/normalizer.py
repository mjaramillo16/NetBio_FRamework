import pandas as pd
import os

def harmonize_and_aggregate(df, mapper=None):
    """
    Aplica o mapeamento de Sondas para Genes e agrega duplicatas.
    Inclui diagnóstico visual e limpeza rigorosa para compatibilidade com o KEGG.
    """
    print("[NORMALIZER] Iniciando harmonização...")
    
    # 1. Diagnóstico Inicial
    if df is None:
        print(" Erro: O DataFrame de entrada é None.")
        return None

    print(f"   > Dimensão Original: {df.shape[0]} linhas x {df.shape[1]} amostras")
    
    # --- DEBUG VISUAL: Mostrar o que temos ---
    print(f"   > Exemplo IDs Matriz (Top 3): {list(df.index[:3])}")
    if mapper:
        print(f"   > Exemplo IDs Mapa (Top 3):   {list(mapper.keys())[:3]}")
    # ----------------------------------------

    # Caso 1: Nenhum mapa fornecido (Assumimos que já são genes)
    if mapper is None:
        print("   > Nenhum mapa fornecido. Assumindo que os IDs já são genes.")
        # Limpeza preventiva mesmo sem mapa
        df.index = df.index.astype(str).str.upper().str.strip()
        df = df.groupby(level=0).mean()
        return df

    # 2. Verificar Interseção
    original_ids = set(df.index)
    mapped_ids = set(mapper.keys())
    intersection = original_ids.intersection(mapped_ids)
    
    # --- LÓGICA DE AUTO-RECUPERAÇÃO ---
    if len(intersection) == 0:
        print("  AVISO: Nenhuma intersecção direta encontrada.")
        
        sample_id_matrix = str(list(df.index)[0])
        print(f" Diagnóstico: Matriz exemplo = '{sample_id_matrix}'")

        # Teste A: Será que a matriz JÁ SÃO Genes? (Ex: "INS", "TP53")
        if "_" not in sample_id_matrix and not sample_id_matrix.isdigit():
             print("HIPÓTESE: A matriz parece já estar usando Gene Symbols.")
             print("      -> Ignorando o mapa GPL e padronizando IDs originais.")
             
             # --- LIMPEZA PARA TOPOLOGIA (Crucial para a tese) ---
             df.index = df.index.astype(str).str.upper().str.strip()
             df = df[df.index.notna() & (df.index != "NAN") & (df.index != "")]
             
             df = df.groupby(level=0).mean()
             return df
        
        print("  ERRO CRÍTICO: Não foi possível harmonizar IDs automaticamente.")
        return None

    print(f"   > Sondas no Mapa: {len(mapped_ids)}")
    print(f"   > Interseção (Sondas úteis): {len(intersection)}")

    # 3. Mapeamento
    df.index = df.index.map(mapper)
    
    # 4. LIMPEZA RIGOROSA PARA COMPATIBILIDADE COM KEGG
    # Transformamos tudo para Maiúsculas e removemos espaços (ex: " akt1 " -> "AKT1")
    df.index = df.index.astype(str).str.upper().str.strip()
    
    # Remove linhas que não foram mapeadas ou que resultaram em nomes inválidos
    df = df[df.index.notna() & (df.index != "NAN") & (df.index != "") & (df.index != "NONE")]
    
    # 5. Agregação (Mean)
    genes_before = df.shape[0]
    df = df.groupby(level=0).mean()
    genes_after = df.shape[0]
    
    print(f"   > Agregação: Reduzido de {genes_before} sondas para {genes_after} genes únicos.")
    print(f"   > Padronização: Todos os Gene Symbols convertidos para UPPERCASE (padrão KEGG).")
    
    return df

def save_as_parquet(df, output_path):
    if df is None:
        print("  Erro: Tentativa de salvar DataFrame vazio (None).")
        return False

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Transpõe para o formato (Amostras, Genes) exigido pelas Neural ODEs
        df_transposed = df.T
        df_transposed.to_parquet(output_path)
        print(f"   Arquivo salvo com sucesso: {output_path}")
        print(f"      Formato Final: {df_transposed.shape} (Amostras, Genes)")
        return True
    except Exception as e:
        print(f"    Erro ao salvar parquet: {e}")
        return False