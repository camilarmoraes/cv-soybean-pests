# -*- coding: utf-8 -*-
"""
Renomeação em lote de arquivos de imagem.

Este script renomeia todos os arquivos dentro de um diretório, aplicando
um padrão consistente: {prefixo}_{número_sequencial}.{extensão}

Útil para resolver conflitos de nomes ao unificar múltiplas bases de dados
que possuem arquivos com nomes duplicados.

Uso:
    Configure DIR_NAME_CHANGE no .env ou passe via CLI
    $ python change_file_name.py --dir ./data/images --prefix septoria_leaf
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO — ajuste via .env ou argumentos CLI
# ============================================================
TARGET_DIR = os.getenv("DIR_NAME_CHANGE", "./data/raw")
DEFAULT_PREFIX = "image"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def rename_files(directory: str, prefix: str = DEFAULT_PREFIX,
                 start_index: int = 1, dry_run: bool = False):
    """
    Renomeia todos os arquivos de imagem em um diretório com padrão sequencial.

    Args:
        directory: Caminho do diretório com os arquivos.
        prefix: Prefixo para os novos nomes (ex: 'septoria_leaf_dataset').
        start_index: Índice inicial da numeração.
        dry_run: Se True, apenas exibe as mudanças sem executar.

    Returns:
        Número de arquivos renomeados.
    """
    if not os.path.isdir(directory):
        print(f"[ERRO] Diretório não encontrado: {directory}")
        return 0

    files = sorted([
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not files:
        print(f"[AVISO] Nenhum arquivo de imagem encontrado em: {directory}")
        return 0

    renamed_count = 0
    for i, filename in enumerate(files, start=start_index):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{prefix}_{i:05d}{ext}"

        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        if old_path == new_path:
            continue

        if dry_run:
            print(f"  [DRY-RUN] {filename} → {new_name}")
        else:
            os.rename(old_path, new_path)
            renamed_count += 1

    action = "simulados" if dry_run else "renomeados"
    print(f"\n[OK] {renamed_count} arquivos {action} em: {directory}")
    print(f"     Padrão: {prefix}_XXXXX{ext}")
    return renamed_count


def main():
    parser = argparse.ArgumentParser(
        description="Renomeação em lote de arquivos de imagem"
    )
    parser.add_argument("--dir", default=TARGET_DIR,
                        help="Diretório com os arquivos a renomear")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="Prefixo para os novos nomes (ex: 'septoria_leaf_dataset')")
    parser.add_argument("--start", type=int, default=1,
                        help="Índice inicial da numeração (padrão: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula a renomeação sem executar")
    args = parser.parse_args()

    if args.dry_run:
        print("[MODO DRY-RUN] Nenhum arquivo será modificado.\n")

    rename_files(args.dir, args.prefix, args.start, args.dry_run)


if __name__ == "__main__":
    main()
