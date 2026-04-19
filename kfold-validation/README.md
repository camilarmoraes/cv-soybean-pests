# Validação Cruzada K-Fold

Notebooks para validação cruzada com K-Fold (K=10) dos 4 modelos CNN.

## Metodologia

- **K=10** folds, usando `KFold` do scikit-learn
- Cada fold: treinamento com 9 partes, validação com 1 parte
- Métricas agregadas: acurácia média ± desvio padrão entre os folds
- Transfer learning com pesos pré-treinados (PlantVillage)
- Hiperparâmetros ótimos do Bayesian Search

## Notebooks

| Notebook | Modelo |
|---|---|
| `resnet101_kfold.ipynb` | ResNet-101 |
| `densenet201_kfold.ipynb` | DenseNet-201 |
| `vgg19_kfold.ipynb` | VGG-19 |
| `mobilenetv3_kfold.ipynb` | MobileNetV3-Large |

## Uso

1. Configure o `.env` com os caminhos do dataset e modelos pré-treinados
2. Execute o notebook desejado
3. Os resultados (acurácia por fold) são apresentados ao final
