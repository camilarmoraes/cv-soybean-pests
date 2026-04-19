# -*- coding: utf-8 -*-
"""
Gerador de plano de pulverização a partir de shapefile classificado.

Este script lê o shapefile gerado pela inferência e cria áreas de
pulverização (buffer zones) ao redor dos tiles classificados como doentes,
gerando um novo shapefile com as zonas de aplicação de defensivos.


Uso:
    $ python spray_plan_generator.py --input classification.shp --output spray_plan.shp
"""

import os
import argparse
import geopandas as gpd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÃO
# ============================================================
OUTPUT_DIR = os.getenv("ORTHOMOSAIC_OUTPUT_DIR", "./results")
BUFFER_DISTANCE = 5.0  # Distância do buffer em metros (ajuste conforme CRS)
MIN_PROBABILITY = 0.7  # Probabilidade mínima para considerar doente


def generate_spray_plan(input_shapefile: str, output_shapefile: str,
                        buffer_distance: float = BUFFER_DISTANCE,
                        min_probability: float = MIN_PROBABILITY):
    """
    Gera um plano de pulverização baseado no shapefile classificado.

    Args:
        input_shapefile: Caminho do shapefile com classificação.
        output_shapefile: Caminho de saída para o shapefile do plano.
        buffer_distance: Distância do buffer (metros).
        min_probability: Probabilidade mínima para classificar como doente.
    """
    gdf = gpd.read_file(input_shapefile)
    print(f"Total de tiles lidos: {len(gdf)}")

    # Filtrar tiles classificados como doentes com alta probabilidade
    disease_tiles = gdf[
        (gdf["class"] == 0) & (gdf["prob_disease"] >= min_probability)
    ].copy()
    print(f"Tiles doentes (prob >= {min_probability}): {len(disease_tiles)}")

    if len(disease_tiles) == 0:
        print("[INFO] Nenhuma área de pulverização necessária.")
        return

    # Criar buffer (zona de pulverização) ao redor de cada tile doente
    disease_tiles["geometry"] = disease_tiles["geometry"].buffer(buffer_distance)

    # Unir buffers sobrepostos
    spray_zone = disease_tiles.dissolve()
    spray_zone = spray_zone.explode(index_parts=False).reset_index(drop=True)

    # Calcular área de cada zona
    spray_zone["area_m2"] = spray_zone.geometry.area
    spray_zone["area_ha"] = spray_zone["area_m2"] / 10000

    # Salvar shapefile do plano de pulverização
    os.makedirs(os.path.dirname(output_shapefile), exist_ok=True)
    spray_zone.to_file(output_shapefile)

    total_area_ha = spray_zone["area_ha"].sum()
    print(f"\nPlano de pulverização gerado:")
    print(f"  Zonas de aplicação: {len(spray_zone)}")
    print(f"  Área total: {total_area_ha:.4f} ha")
    print(f"  Shapefile salvo em: {output_shapefile}")


def main():
    parser = argparse.ArgumentParser(description="Gerador de plano de pulverização")
    parser.add_argument("--input", default=os.path.join(OUTPUT_DIR, "classification.shp"),
                        help="Shapefile de classificação de entrada")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "spray_plan.shp"),
                        help="Shapefile de saída com plano de pulverização")
    parser.add_argument("--buffer", type=float, default=BUFFER_DISTANCE,
                        help="Distância do buffer em metros")
    parser.add_argument("--min-prob", type=float, default=MIN_PROBABILITY,
                        help="Probabilidade mínima para considerar doente")
    args = parser.parse_args()

    generate_spray_plan(args.input, args.output, args.buffer, args.min_prob)


if __name__ == "__main__":
    main()
