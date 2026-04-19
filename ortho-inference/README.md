# Inferência em Ortomosaico

Scripts para aplicação dos modelos treinados em ortofotos georreferenciadas obtidas por VANT (drone), com geração de shapefiles e planos de pulverização.

> **Nota:** O melhor modelo para inferência foi o **DenseNet-201**, que obteve a maior acurácia geral nos experimentos.

## Scripts Disponíveis

| Script | Descrição |
|---|---|
| `orthomosaic_inference.py` | Inferência básica com geração de shapefile classificado |
| `orthomosaic_ndvi_filter.py` | Inferência com filtro NDVI e substituição de tiles |
| `spray_plan_generator.py` | Geração de plano de pulverização a partir do shapefile |
| `tile_heatmap.py` | Grad-CAM em tiles individuais do ortomosaico |

## Uso

### 1. Inferência básica
```bash
python orthomosaic_inference.py \
    --orthomosaic /caminho/ortomosaico.tif \
    --model-path /caminho/modelo.pt \
    --model-name densenet201 \
    --output ./results/classification.shp
```

### 2. Inferência com filtro NDVI
```bash
python orthomosaic_ndvi_filter.py \
    --orthomosaic /caminho/ortomosaico.tif \
    --model-path /caminho/modelo.pt \
    --model-name densenet201 \
    --ndvi-threshold 0.35
```

### 3. Gerar plano de pulverização
```bash
python spray_plan_generator.py \
    --input ./results/classification.shp \
    --output ./results/spray_plan.shp \
    --buffer 5.0
```

### 4. Gerar mapas de calor (Grad-CAM) em tiles
```bash
python tile_heatmap.py \
    --input-dir ./tiles \
    --model-path /caminho/modelo.pt \
    --model-name densenet201
```

## Modelos Suportados

Todos os scripts suportam os 4 modelos treinados:
- `resnet101` — ResNet-101
- `densenet201` — DenseNet-201 **(recomendado)**
- `vgg19` — VGG-19
- `mobilenetv3` — MobileNetV3-Large

## Configuração via `.env`

Os caminhos podem ser configurados via arquivo `.env` na raiz do projeto:
```
ORTHOMOSAIC_PATH=/caminho/para/ortomosaico.tif
TRAINED_MODEL_PATH=/caminho/para/modelo.pt
MODEL_NAME=densenet201
ORTHOMOSAIC_OUTPUT_DIR=./results
AVULSA_IMAGES_PATH=/caminho/para/imagens-avulsas
```
