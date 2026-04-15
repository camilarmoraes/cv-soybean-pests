# -*- coding: utf-8 -*-
"""
Redimensionamento de imagens para 224x224 pixels.

Este script percorre todas as imagens de um dataset organizado em subpastas
(ex: healthy/, diseases/) e redimensiona cada imagem para o tamanho padrão
de entrada das CNNs utilizadas neste projeto (224x224 pixels).

Baseado no código original: CODES/PROCESSAMENTO-IMAGEM/01-resolucao-imagem.py

Uso:
    Ajuste os caminhos INPUT_DIR e OUTPUT_DIR abaixo ou configure via .env
    $ python resize_images.py
"""

import os
import cv2
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO — ajuste estes caminhos conforme necessário
# ============================================================
INPUT_DIR = os.getenv("RAW_DATASET_PATH", "./data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DATASET_PATH", "./data/resized")
TARGET_SIZE = (224, 224)


def resize_images(input_dir: str, output_dir: str, target_size: tuple = TARGET_SIZE):
    """
    Redimensiona todas as imagens dentro de input_dir para target_size
    e salva em output_dir mantendo a estrutura de subpastas.

    Args:
        input_dir: Diretório raiz contendo subpastas com imagens.
        output_dir: Diretório de saída.
        target_size: Tupla (largura, altura) do tamanho final.
    """
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)

        for idx, filename in enumerate(os.listdir(class_input_path)):
            filepath = os.path.join(class_input_path, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"[AVISO] Não foi possível ler: {filepath}")
                continue

            img_resized = cv2.resize(img, target_size)
            output_filename = f"{class_name}_{idx:05d}.jpg"
            output_path = os.path.join(class_output_path, output_filename)
            cv2.imwrite(output_path, img_resized)

        print(f"[OK] {class_name}: {idx + 1} imagens redimensionadas para {target_size}")

    print(f"\nImagens salvas em: {output_dir}")


if __name__ == "__main__":
    resize_images(INPUT_DIR, OUTPUT_DIR)
