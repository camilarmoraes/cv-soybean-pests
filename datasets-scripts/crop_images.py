# -*- coding: utf-8 -*-
"""
Recorte central de imagens (center crop).

Este script aplica um recorte central de tamanho configurável em todas as
imagens de um dataset, útil para padronizar imagens que possuem bordas
irrelevantes ou tamanhos variados.

Baseado no código original: CODES/DATABASE/cropped-25.py

Uso:
    Ajuste os caminhos INPUT_DIR e OUTPUT_DIR abaixo ou configure via .env
    $ python crop_images.py
"""

import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO — ajuste conforme necessário
# ============================================================
INPUT_DIR = os.getenv("RAW_DATASET_PATH", "./data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DATASET_PATH", "./data/cropped")
CROP_PERCENT = 0.25  # Porcentagem da borda a ser removida (25%)


def center_crop(image: Image.Image, crop_percent: float = CROP_PERCENT) -> Image.Image:
    """
    Aplica um recorte central na imagem, removendo crop_percent de cada borda.

    Args:
        image: Imagem PIL.
        crop_percent: Porcentagem da borda a remover (0.25 = 25% de cada lado).

    Returns:
        Imagem recortada.
    """
    width, height = image.size
    left = int(width * crop_percent)
    top = int(height * crop_percent)
    right = int(width * (1 - crop_percent))
    bottom = int(height * (1 - crop_percent))
    return image.crop((left, top, right, bottom))


def crop_dataset(input_dir: str, output_dir: str, crop_percent: float = CROP_PERCENT):
    """
    Aplica center crop em todas as imagens do dataset.

    Args:
        input_dir: Diretório de entrada com subpastas por classe.
        output_dir: Diretório de saída.
        crop_percent: Porcentagem da borda a remover.
    """
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)
        count = 0

        for filename in os.listdir(class_input_path):
            filepath = os.path.join(class_input_path, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                img_cropped = center_crop(img, crop_percent)
                output_path = os.path.join(class_output_path, filename)
                img_cropped.save(output_path)
                count += 1
            except Exception as e:
                print(f"[ERRO] {filepath}: {e}")

        print(f"[OK] {class_name}: {count} imagens recortadas (crop={crop_percent*100:.0f}%)")

    print(f"\nImagens salvas em: {output_dir}")


if __name__ == "__main__":
    crop_dataset(INPUT_DIR, OUTPUT_DIR)
