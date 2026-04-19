# Análise e Visualização de Resultados

Análise de interpretabilidade dos modelos, incluindo geração de mapas de calor (Grad-CAM) para entender quais regiões das folhas o modelo está focando.

## Notebooks

| Notebook | Descrição |
|---|---|
| `grad_cam_heatmaps.ipynb` | Geração de mapas de calor Grad-CAM para todos os 4 modelos |

## Grad-CAM (Gradient-weighted Class Activation Mapping)

O Grad-CAM utiliza os gradientes fluindo para a última camada convolucional para produzir um mapa de localização que destaca as regiões importantes na imagem para predição do conceito-alvo.

### Camadas alvo por modelo

| Modelo | Camada Alvo |
|---|---|
| ResNet-101 | `layer4[-1].conv3` |
| DenseNet-201 | `features.denseblock4.denselayer32.conv2` |
| VGG-19 | `features[-1]` |
| MobileNetV3 | `features[-1]` |
