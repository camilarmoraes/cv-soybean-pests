# -*- coding: utf-8 -*-
"""
Divisão automática de dataset em train/val/test.

Este script percorre um diretório raiz contendo subpastas de datasets
(cada uma com subpastas por classe, ex: Healthy/, Diseases/) e divide
automaticamente as imagens em conjuntos de treino, validação e teste.

O split padrão é 75% treino / 15% validação / 10% teste.

Uso:
    Configure as variáveis ROOT_DIR_SPLIT e TARGET_DIR_SPLIT no .env
    $ python automatizate_split_dataset.py
"""

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO — ajuste via .env ou argumentos CLI
# ============================================================
ROOT_DIR = os.getenv("ROOT_DIR_SPLIT", "./data/raw")
TARGET_DIR = os.getenv("TARGET_DIR_SPLIT", "./data/split")

# Proporções de divisão
TRAIN_PERCENT = 0.75
VAL_PERCENT = 0.15
TEST_PERCENT = 0.10

# Mapeamento de nomes de classe (normalização)
# Converte nomes originais das pastas para nomes padronizados
CLASS_MAPPING = {
    "Healthy": "healthy",
    "healthy": "healthy",
    "Diseases": "diseases",
    "diseases": "diseases",
}


def create_target_dirs(target_dir: str, class_names: list):
    """
    Cria a estrutura de diretórios train/val/test com subpastas por classe.

    Args:
        target_dir: Diretório raiz de destino.
        class_names: Lista de nomes de classe (ex: ['healthy', 'diseases']).
    """
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            dir_path = os.path.join(target_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)

    print(f"Estrutura de diretórios criada em: {target_dir}")


def split_files(files: list, train_pct: float = TRAIN_PERCENT,
                test_pct: float = TEST_PERCENT, random_state: int = 42):
    """
    Divide uma lista de arquivos em train, val e test.

    Args:
        files: Lista de nomes de arquivos.
        train_pct: Proporção para treino (padrão: 0.75).
        test_pct: Proporção para teste (padrão: 0.10).
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla (train_files, val_files, test_files).
    """
    temp_size = 1 - train_pct
    test_ratio = test_pct / temp_size

    train_files, temp_files = train_test_split(
        files, test_size=temp_size, random_state=random_state
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio, random_state=random_state
    )

    return train_files, val_files, test_files


def copy_files_to_split(file_list: list, source_dir: str, dest_dir: str):
    """
    Copia uma lista de arquivos de source_dir para dest_dir.

    Args:
        file_list: Lista de nomes de arquivos.
        source_dir: Diretório de origem.
        dest_dir: Diretório de destino.

    Returns:
        Número de arquivos copiados com sucesso.
    """
    copied = 0
    for filename in file_list:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"[ERRO] {src} → {dst}: {e}")
    return copied


def process_dataset(root_dir: str, target_dir: str,
                    train_pct: float = TRAIN_PERCENT,
                    test_pct: float = TEST_PERCENT):
    """
    Processa o dataset completo: percorre as subpastas, divide e copia.

    Espera a estrutura:
        root_dir/
        ├── dataset1/
        │   ├── Healthy/
        │   └── Diseases/
        └── dataset2/
            ├── Healthy/
            └── Diseases/

    Args:
        root_dir: Diretório raiz com os datasets originais.
        target_dir: Diretório de destino para o split.
        train_pct: Proporção de treino.
        test_pct: Proporção de teste.
    """
    # Identificar classes únicas
    unique_classes = set(CLASS_MAPPING.values())
    create_target_dirs(target_dir, list(unique_classes))

    total_copied = {"train": 0, "val": 0, "test": 0}

    for dataset_name in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessando: {dataset_name}/")

        for class_folder in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Normalizar nome da classe
            normalized_class = CLASS_MAPPING.get(class_folder, class_folder.lower())

            # Listar arquivos de imagem
            files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]

            if not files:
                print(f"  [AVISO] {class_folder}: nenhuma imagem encontrada")
                continue

            # Dividir arquivos
            train_files, val_files, test_files = split_files(
                files, train_pct, test_pct
            )

            # Copiar para os diretórios de destino
            splits = {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }

            for split_name, split_files_list in splits.items():
                dest = os.path.join(target_dir, split_name, normalized_class)
                copied = copy_files_to_split(split_files_list, class_path, dest)
                total_copied[split_name] += copied

            print(f"  {class_folder} → {normalized_class}: "
                  f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Resumo
    print(f"\n{'='*50}")
    print(f"RESUMO DO SPLIT ({train_pct*100:.0f}/{(1-train_pct-test_pct)*100:.0f}/{test_pct*100:.0f})")
    print(f"{'='*50}")
    for split_name, count in total_copied.items():
        print(f"  {split_name}: {count} imagens")
    print(f"  Total: {sum(total_copied.values())} imagens")
    print(f"\nDataset salvo em: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Divisão automática de dataset em train/val/test"
    )
    parser.add_argument("--root-dir", default=ROOT_DIR,
                        help="Diretório raiz com os datasets originais")
    parser.add_argument("--target-dir", default=TARGET_DIR,
                        help="Diretório de destino para o split")
    parser.add_argument("--train-pct", type=float, default=TRAIN_PERCENT,
                        help="Proporção de treino (padrão: 0.75)")
    parser.add_argument("--test-pct", type=float, default=TEST_PERCENT,
                        help="Proporção de teste (padrão: 0.10)")
    args = parser.parse_args()

    process_dataset(args.root_dir, args.target_dir, args.train_pct, args.test_pct)


if __name__ == "__main__":
    main()
