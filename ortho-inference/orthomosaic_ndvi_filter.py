# -*- coding: utf-8 -*-
"""
Inferência em ortomosaico com filtro NDVI e substituição de tiles.

Este script estende a inferência básica adicionando:
1. Filtro NDVI para remover tiles sem vegetação (requer banda NIR)
2. Substituição aleatória de tiles por imagens avulsas de teste

# NOTA: O melhor modelo para inferência em ortomosaico foi o DenseNet-201,
# que obteve a melhor acurácia geral nos experimentos do TCC.

Baseado em: ORTOMOSAIC/codes/13_inferencia.py e 12_inferencia_removendo_terreno.py

Uso:
    $ python orthomosaic_ndvi_filter.py --model-name densenet201
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Polygon
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Importa funções do script principal de inferência
from orthomosaic_inference import load_model, load_orthomosaic, apply_model_with_probabilities

load_dotenv()

# ============================================================
# CONFIGURAÇÃO
# ============================================================
ORTHOMOSAIC_PATH = os.getenv("ORTHOMOSAIC_PATH", "./data/ortomosaico.tif")
MODEL_PATH = os.getenv("TRAINED_MODEL_PATH", "./models/best_model.pt")
AVULSA_FOLDER = os.getenv("AVULSA_IMAGES_PATH", "./data/imagens-avulsas")
OUTPUT_DIR = os.getenv("ORTHOMOSAIC_OUTPUT_DIR", "./results")
MODEL_NAME = os.getenv("MODEL_NAME", "densenet201")
NDVI_THRESHOLD = 0.35
REPLACEMENT_PROB = 0.1  # Probabilidade de substituição por imagem avulsa
TILE_SIZE = 224


def calculate_ndvi(red_band, nir_band):
    """Calcula o NDVI (Normalized Difference Vegetation Index)."""
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    return (nir - red) / (nir + red + 1e-10)


def has_vegetation(tile, ndvi_threshold=NDVI_THRESHOLD):
    """
    Verifica se o tile contém vegetação com base no NDVI.

    Args:
        tile: Array (C, H, W) com pelo menos 4 bandas (R, G, B, NIR).
        ndvi_threshold: Limiar mínimo de NDVI médio.

    Returns:
        True se o NDVI médio do tile supera o limiar.
    """
    if tile.shape[0] < 4:
        raise ValueError("O ortomosaico deve ter pelo menos 4 bandas (R, G, B, NIR).")

    red_band = tile[0, :, :]
    nir_band = tile[3, :, :]
    ndvi = calculate_ndvi(red_band, nir_band)
    return np.nanmean(ndvi) > ndvi_threshold


def load_avulsa_images(folder_path):
    """Carrega imagens avulsas de uma pasta para substituição de tiles."""
    tiles = []
    if not os.path.exists(folder_path):
        print(f"[AVISO] Pasta de imagens avulsas não encontrada: {folder_path}")
        return tiles

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, filename)
            image = plt.imread(path)
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            tiles.append(image)
    print(f"Carregadas {len(tiles)} imagens avulsas de {folder_path}")
    return tiles


def split_with_ndvi_filter(orthomosaic, meta, avulsa_tiles, tile_size=TILE_SIZE,
                           nodata_threshold=0.5, replacement_prob=REPLACEMENT_PROB,
                           ndvi_threshold=NDVI_THRESHOLD):
    """
    Divide o ortomosaico em tiles com filtro NDVI e substituição opcional.

    Returns:
        Tupla (tiles, windows, replaced_info).
    """
    _, height, width = orthomosaic.shape
    tiles, windows, replaced_info = [], [], []
    has_alpha = orthomosaic.shape[0] == 4

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            if y + tile_size > height or x + tile_size > width:
                continue

            window = Window(x, y, tile_size, tile_size)
            tile = orthomosaic[:, y:y+tile_size, x:x+tile_size]

            # Filtro NDVI — descarta tiles sem vegetação
            if not has_vegetation(tile, ndvi_threshold):
                continue

            # Verificação de dados válidos
            if has_alpha:
                valid_pixels = np.sum(tile[3, :, :] > 0)
            elif meta["nodata"] is not None:
                mask = np.all(tile == meta["nodata"], axis=0)
                valid_pixels = np.sum(~mask)
            else:
                valid_pixels = tile_size ** 2

            if valid_pixels / (tile_size ** 2) < nodata_threshold:
                continue

            # Substituição aleatória por imagem avulsa
            is_replaced = False
            if np.random.rand() < replacement_prob and avulsa_tiles:
                avulsa_tile = avulsa_tiles[np.random.randint(0, len(avulsa_tiles))]
                tile = np.transpose(avulsa_tile, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                is_replaced = True

                transform = meta["transform"]
                x_ul, y_ul = transform * (x, y)
                x_br, y_br = transform * (x + tile_size, y + tile_size)
                replaced_info.append({
                    "window": window,
                    "coordinates": {"ul": (x_ul, y_ul), "br": (x_br, y_br)},
                })

            tiles.append(tile)
            windows.append(window)

    return tiles, windows, replaced_info


def create_shapefile_with_replacement(predictions, windows, meta, output_path, replaced_info):
    """Cria shapefile incluindo informação de tiles substituídos."""
    features = []
    for (predicted_class, probabilities), window in zip(predictions, windows):
        col_off, row_off = window.col_off, window.row_off
        width, height = window.width, window.height

        x_min, y_min = meta["transform"] * (col_off, row_off)
        x_max, y_max = meta["transform"] * (col_off + width, row_off + height)

        polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
        ])

        is_replaced = any(info["window"] == window for info in replaced_info)

        features.append({
            "geometry": polygon,
            "class": int(predicted_class),
            "prob_disease": float(probabilities[0]),
            "prob_healthy": float(probabilities[1]),
            "replaced": int(is_replaced),
        })

    gdf = gpd.GeoDataFrame(features, crs=meta["crs"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path)
    print(f"Shapefile salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inferência em ortomosaico com filtro NDVI")
    parser.add_argument("--orthomosaic", default=ORTHOMOSAIC_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--avulsa-folder", default=AVULSA_FOLDER)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "classification_ndvi.shp"))
    parser.add_argument("--ndvi-threshold", type=float, default=NDVI_THRESHOLD)
    parser.add_argument("--replacement-prob", type=float, default=REPLACEMENT_PROB)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Modelo: {args.model_name}")

    orthomosaic, meta = load_orthomosaic(args.orthomosaic)
    avulsa_tiles = load_avulsa_images(args.avulsa_folder)

    tiles, windows, replaced_info = split_with_ndvi_filter(
        orthomosaic, meta, avulsa_tiles,
        ndvi_threshold=args.ndvi_threshold,
        replacement_prob=args.replacement_prob,
    )
    print(f"Tiles válidos: {len(tiles)} | Substituídos: {len(replaced_info)}")

    model = load_model(args.model_name, args.model_path, device)
    predictions = [apply_model_with_probabilities(tile, model, device) for tile in tiles]
    create_shapefile_with_replacement(predictions, windows, meta, args.output, replaced_info)
    print("Inferência com filtro NDVI finalizada!")


if __name__ == "__main__":
    main()
