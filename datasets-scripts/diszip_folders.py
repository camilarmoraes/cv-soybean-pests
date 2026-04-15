# -*- coding: utf-8 -*-
"""
Extração em lote de arquivos ZIP.

Este script extrai todos os arquivos .zip de um diretório de origem
para um diretório de destino, criando subpastas com o nome de cada
arquivo ZIP (sem a extensão).

Uso:
    Configure ZIP_FOLDER e DISZIP_FOLDER no .env ou passe via CLI
    $ python diszip_folders.py --source ./data/zips --dest ./data/extracted
"""

import os
import zipfile
import argparse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO — ajuste via .env ou argumentos CLI
# ============================================================
SOURCE_DIR = os.getenv("ZIP_FOLDER", "./data/zips")
DEST_DIR = os.getenv("DISZIP_FOLDER", "./data/extracted")


def extract_zip(zip_path: str, dest_path: str) -> bool:
    """
    Extrai um arquivo ZIP para o diretório de destino.

    Args:
        zip_path: Caminho completo do arquivo .zip.
        dest_path: Diretório de destino para extração.

    Returns:
        True se extraído com sucesso, False caso contrário.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        return True
    except zipfile.BadZipFile:
        print(f"[ERRO] Arquivo ZIP corrompido: {zip_path}")
        return False
    except Exception as e:
        print(f"[ERRO] Falha ao extrair {zip_path}: {e}")
        return False


def extract_all_zips(source_dir: str, dest_dir: str):
    """
    Extrai todos os arquivos .zip de source_dir para dest_dir.

    Cada ZIP é extraído em uma subpasta com seu nome (sem extensão).

    Args:
        source_dir: Diretório contendo os arquivos .zip.
        dest_dir: Diretório raiz de destino.
    """
    if not os.path.isdir(source_dir):
        print(f"[ERRO] Diretório de origem não encontrado: {source_dir}")
        return

    zip_files = sorted([
        f for f in os.listdir(source_dir)
        if f.lower().endswith(".zip")
    ])

    if not zip_files:
        print(f"[AVISO] Nenhum arquivo .zip encontrado em: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    success_count = 0

    print(f"Extraindo {len(zip_files)} arquivos ZIP...")
    print(f"  Origem:  {source_dir}")
    print(f"  Destino: {dest_dir}\n")

    for filename in zip_files:
        zip_path = os.path.join(source_dir, filename)
        folder_name = os.path.splitext(filename)[0]
        extract_dest = os.path.join(dest_dir, folder_name)

        os.makedirs(extract_dest, exist_ok=True)

        if extract_zip(zip_path, extract_dest):
            num_files = sum(len(files) for _, _, files in os.walk(extract_dest))
            print(f"  [OK] {filename} → {folder_name}/ ({num_files} arquivos)")
            success_count += 1
        else:
            print(f"  [FALHA] {filename}")

    print(f"\n{'='*40}")
    print(f"Extração finalizada: {success_count}/{len(zip_files)} arquivos")


def main():
    parser = argparse.ArgumentParser(
        description="Extração em lote de arquivos ZIP"
    )
    parser.add_argument("--source", default=SOURCE_DIR,
                        help="Diretório contendo os arquivos .zip")
    parser.add_argument("--dest", default=DEST_DIR,
                        help="Diretório de destino para extração")
    args = parser.parse_args()

    extract_all_zips(args.source, args.dest)


if __name__ == "__main__":
    main()
