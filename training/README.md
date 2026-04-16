# Treinamento dos Modelos

Notebooks Jupyter para fine-tuning dos 4 modelos CNN utilizados na classificação binária de folhas de soja (saudável vs. doente).

## Modelos

| Notebook | Modelo | Pesos base | Classificador |
|---|---|---|---|
| `resnet101_training.ipynb` | ResNet-101 | `ResNet_101_ImageNet_plant-model-84.pth` | FC Custom |
| `densenet201_training.ipynb` | DenseNet-201 | `densenet201-model-95.pth` | FC Custom |
| `vgg19_training.ipynb` | VGG-19 | `VggNet19-model-96.pth` | FC Custom |
| `mobilenetv3_training.ipynb` | MobileNetV3-Large | `mobilenet_v3_large-model-84.pth` | FC Custom |

## Metodologia

Cada notebook segue o mesmo pipeline:

1. **Configuração** — imports, device, caminhos configuráveis
2. **Data Augmentation** — RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomVerticalFlip, RandomPerspective
3. **Carregamento** — ImageFolder via torchvision com 3 splits (train/val/test)
4. **Transfer Learning** — Feature extraction com congelamento de camadas, seguido de classificador customizado com os **hiperparâmetros ótimos** encontrados via Bayesian Search (Optuna)
5. **Treinamento** — PyTorch Ignite (Engine, Metrics, EarlyStopping, ModelCheckpoint)
6. **Avaliação** — Métricas no conjunto de teste
7. **Visualização** — Curvas de loss e accuracy