# -*- coding: utf-8 -*-
"""
Inferência em ortomosaico com geração de shapefile.

Este script divide um ortomosaico georreferenciado em tiles de 224x224 pixels,
aplica um modelo CNN treinado para classificação binária (saudável/doente),
e gera um shapefile georreferenciado com as probabilidades de cada classe.

O script suporta qualquer um dos 4 modelos treinados no estudo:
- ResNet-101, DenseNet-201, VGG-19, MobileNetV3-Large

# NOTA: O melhor modelo para inferência em ortomosaico foi o DenseNet-201,
# que obteve a melhor acurácia geral (93.75%) nos experimentos do TCC.

Baseado em: ORTOMOSAIC/codes/06_inferencia_probabilidades.py

Uso:
    $ python orthomosaic_inference.py
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
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO
# ============================================================
ORTHOMOSAIC_PATH = os.getenv("ORTHOMOSAIC_PATH", "./data/ortomosaico.tif")
MODEL_PATH = os.getenv("TRAINED_MODEL_PATH", "./models/best_model.pt")
OUTPUT_DIR = os.getenv("ORTHOMOSAIC_OUTPUT_DIR", "./results")
TILE_SIZE = 224
NODATA_THRESHOLD = 0.5  # Mínimo de pixels válidos para considerar o tile

# Escolha do modelo: resnet101, densenet201, vgg19, mobilenetv3
# NOTA: O DenseNet-201 foi o melhor modelo no estudo do TCC
MODEL_NAME = os.getenv("MODEL_NAME", "densenet201")

# Hiperparâmetros ótimos encontrados via Bayesian Search para cada modelo
BEST_HYPERPARAMS = {
    "resnet101": {
        "dropout1": 0.4720140263972287,
        "dropout2": 0.21316165290099465,
        "fc1": 256,
        "fc2": 256,
    },
    # O DenseNet-201 obteve o melhor desempenho geral nos experimentos
    "densenet201": {
        "dropout1": 0.3372612925395379,
        "dropout2": 0.2672714454014481,
        "fc1": 512,
        "fc2": 256,
    },
    "vgg19": {
        "dropout1": 0.47514918459654953,
        "dropout2": 0.25083233173075137,
        "fc1": 1024,
        "fc2": 256,
    },
    "mobilenetv3": {
        "dropout1": 0.4730438988122177,
        "dropout2": 0.3254939388545582,
        "fc1": 512,
        "fc2": 512,
    },
}


def load_model(model_name: str, model_path: str, device: torch.device) -> nn.Module:
    """
    Carrega o modelo CNN com a arquitetura e hiperparâmetros corretos.

    Args:
        model_name: Nome do modelo (resnet101, densenet201, vgg19, mobilenetv3).
        model_path: Caminho para os pesos treinados (.pt).
        device: Dispositivo (cuda/cpu).

    Returns:
        Modelo carregado e em modo de avaliação.
    """
    hp = BEST_HYPERPARAMS[model_name]

    if model_name == "resnet101":
        model = models.resnet101(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=hp["dropout1"]),
            nn.Linear(num_features, hp["fc1"]),
            nn.ReLU(),
            nn.Dropout(p=hp["dropout2"]),
            nn.Linear(hp["fc1"], hp["fc2"]),
            nn.ReLU(),
            nn.Linear(hp["fc2"], 2),
        )

    elif model_name == "densenet201":
        model = models.densenet201(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=hp["dropout1"]),
            nn.Linear(num_features, hp["fc1"]),
            nn.ReLU(),
            nn.Dropout(p=hp["dropout2"]),
            nn.Linear(hp["fc1"], hp["fc2"]),
            nn.ReLU(),
            nn.Linear(hp["fc2"], 2),
            nn.Softmax(dim=1),
        )

    elif model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=hp["dropout1"]),
            nn.Linear(num_features, hp["fc1"]),
            nn.ReLU(),
            nn.Dropout(p=hp["dropout2"]),
            nn.Linear(hp["fc1"], hp["fc2"]),
            nn.ReLU(),
            nn.Linear(hp["fc2"], 2),
        )

    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(pretrained=False)
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=hp["dropout1"]),
            nn.Linear(num_features, hp["fc1"]),
            nn.ReLU(),
            nn.Dropout(p=hp["dropout2"]),
            nn.Linear(hp["fc1"], hp["fc2"]),
            nn.ReLU(),
            nn.Linear(hp["fc2"], 2),
            nn.Softmax(dim=1),
        )
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def load_orthomosaic(path: str):
    """Carrega o ortomosaico e seus metadados."""
    with rasterio.open(path) as src:
        orthomosaic = src.read()
        meta = src.meta.copy()
        meta["nodata"] = src.nodata
    return orthomosaic, meta


def split_into_tiles(orthomosaic, meta, tile_size=TILE_SIZE, nodata_threshold=NODATA_THRESHOLD):
    """
    Divide o ortomosaico em tiles, ignorando áreas com muitos pixels inválidos.

    Args:
        orthomosaic: Array numpy (C, H, W) do ortomosaico.
        meta: Metadados do rasterio.
        tile_size: Tamanho do tile em pixels.
        nodata_threshold: Fração mínima de pixels válidos.

    Returns:
        Tupla (tiles, windows) onde tiles é lista de arrays numpy
        e windows é lista de rasterio.windows.Window.
    """
    _, height, width = orthomosaic.shape
    tiles = []
    windows = []
    has_alpha = orthomosaic.shape[0] == 4

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            if y + tile_size > height or x + tile_size > width:
                continue

            window = Window(x, y, tile_size, tile_size)
            tile = orthomosaic[:, y:y+tile_size, x:x+tile_size]

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

            tiles.append(tile)
            windows.append(window)

    return tiles, windows


def apply_model_with_probabilities(tile, model, device):
    """
    Aplica o modelo no tile e retorna a classe predita e as probabilidades.

    Args:
        tile: Array numpy (C, H, W) do tile.
        model: Modelo PyTorch.
        device: Dispositivo.

    Returns:
        Tupla (classe_predita, probabilidades).
    """
    if tile.ndim == 3 and tile.shape[0] == 4:
        tile = tile[:3]  # Remove canal alfa

    tile = np.transpose(tile, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    tile_image = Image.fromarray((tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8))

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tile_tensor = transform(tile_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tile_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities


def create_shapefile(predictions, windows, meta, output_path):
    """
    Cria um shapefile georreferenciado com os resultados da classificação.

    Args:
        predictions: Lista de tuplas (classe, probabilidades).
        windows: Lista de rasterio.windows.Window.
        meta: Metadados do ortomosaico.
        output_path: Caminho do shapefile de saída.
    """
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

        features.append({
            "geometry": polygon,
            "class": int(predicted_class),
            "prob_disease": float(probabilities[0]),
            "prob_healthy": float(probabilities[1]),
        })

    gdf = gpd.GeoDataFrame(features, crs=meta["crs"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path)
    print(f"Shapefile salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inferência em ortomosaico")
    parser.add_argument("--orthomosaic", default=ORTHOMOSAIC_PATH, help="Caminho do ortomosaico")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Caminho do modelo treinado")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Nome do modelo (resnet101, densenet201, vgg19, mobilenetv3)")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "classification.shp"), help="Caminho de saída do shapefile")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="Tamanho do tile")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {args.model_name}")

    # Carregar ortomosaico
    print("Carregando ortomosaico...")
    orthomosaic, meta = load_orthomosaic(args.orthomosaic)

    # Dividir em tiles
    print("Dividindo em tiles...")
    tiles, windows = split_into_tiles(orthomosaic, meta, args.tile_size)
    print(f"Total de tiles válidos: {len(tiles)}")

    # Carregar modelo
    print("Carregando modelo...")
    model = load_model(args.model_name, args.model_path, device)

    # Classificar tiles
    print("Classificando tiles...")
    predictions = [apply_model_with_probabilities(tile, model, device) for tile in tiles]

    # Gerar shapefile
    create_shapefile(predictions, windows, meta, args.output)
    print("Inferência finalizada!")


if __name__ == "__main__":
    main()
