# src/etl/downloader.py

import requests
import os
import gzip
import shutil
import time
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session():
    """Configura uma sessão HTTP com Retry automático para resiliência."""
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_file(url, local_path):
    """
    Baixa um arquivo com proteção contra falhas (Bulletproof).
    - Verifica cache.
    - Tenta 5 vezes com espera exponencial.
    - Mostra barra de progresso.
    - Descompacta automaticamente .gz se necessário.
    """
    filename = os.path.basename(local_path)
    print(f"\n[DOWNLOAD] Iniciando: {filename}")
    
    # 1. Verificar Cache
    if os.path.exists(local_path):
        # Verifica se o arquivo não está vazio/corrompido (tamanho mínimo 10KB)
        if os.path.getsize(local_path) > 10000:
            print("   > Cache encontrado e válido. Pulando download.")
            return True
        else:
            print("   > Arquivo existente corrompido ou muito pequeno. Baixando novamente.")
            os.remove(local_path)

    # 2. Loop de Tentativas
    max_attempts = 5
    session = get_session()
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"   > Tentativa {attempt}/{max_attempts}...")
            
            # Request com stream
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Nome temporário para o download compactado
                temp_gz = local_path + ".gz"
                
                with tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
                    with open(temp_gz, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
            
            # Se chegou aqui, o download terminou. Agora descompactar.
            print("   > Descompactando...")
            with gzip.open(temp_gz, 'rb') as f_in, open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            # Limpa o arquivo .gz temporário
            if os.path.exists(temp_gz):
                os.remove(temp_gz)
                
            print(f"Sucesso: {filename}")
            return True

        except Exception as e:
            print(f"Erro na tentativa {attempt}: {e}")
            time.sleep(2 ** attempt) # Backoff exponencial (2s, 4s, 8s...)
            
            # Limpeza de arquivos parciais em caso de erro
            if os.path.exists(local_path + ".gz"):
                os.remove(local_path + ".gz")

    print(f"FALHA FATAL: Não foi possível baixar {filename} após {max_attempts} tentativas.")
    return False