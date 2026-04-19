# Busca de Hiperparâmetros (Bayesian Search)

Notebooks para otimização bayesiana de hiperparâmetros usando **Optuna** para cada uma das 4 arquiteturas CNN.

## Hiperparâmetros Otimizados

| Parâmetro | Espaço de Busca |
|---|---|
| `dropout1` | 0.2 — 0.5 |
| `dropout2` | 0.2 — 0.5 |
| `num_neurons_fc1` | {256, 512, 1024, 4096} |
| `num_neurons_fc2` | {256, 512, 1024, 4096} |
| `activation` | {ReLU, LeakyReLU} |
| `batch_size` | {32, 64, 128} |
| `optimizer` | {Adam, SGD} |
| `lr` | 1e-5 — 1e-2 (log-uniform) |
| `momentum` (SGD) | 0.7 — 0.99 |

## Melhores Resultados

| Modelo | Melhor Accuracy (Val) | Optimizer | LR |
|---|---|---|---|
| **ResNet-101** | 0.9853 | SGD | 0.00116 |
| **DenseNet-201** | 0.9375 | — | — |
| **VGG-19** | 0.9780 | Adam | 0.000125 |
| **MobileNetV3** | 0.9890 | Adam | 0.00483 |

## Notebooks

- `resnet101_bayesian_search.ipynb`
- `densenet201_bayesian_search.ipynb`
- `vgg19_bayesian_search.ipynb`
- `mobilenetv3_bayesian_search.ipynb`

## Uso

Cada notebook realiza 10 trials de otimização. Os melhores hiperparâmetros encontrados são utilizados nos notebooks de treinamento final (pasta `training/`).
