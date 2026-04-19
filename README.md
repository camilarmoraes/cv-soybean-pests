Este repositório contém os códigos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) intitulado:  
**"Avaliação da Eficácia de Detecção de Pragas em Culturas de Soja Utilizando Visão Computacional"**.

## 📌 Contexto do Projeto

O trabalho tem como foco a avaliação de modelos de redes neurais convolucionais pré-treinados para a detecção de doenças em plantas de soja.

O processo de treinamento foi realizado com imagens provenientes de bases de dados públicas, nas quais as plantas foram fotografadas em ambientes terrestres (com proximidade a planta). A partir desses modelos treinados, buscou-se avaliar o desempenho deles em um novo cenário: **a inferência sobre imagens capturadas por drones em ambiente aéreo**, considerando fatores como **altura de voo, angulação da câmera e velocidade da aeronave**.

## 🧪 Objetivos

1. **Preparação do dataset** — Integração de bases públicas + coleta própria, split 75/15/10
2. **Transfer Learning** — Fine-tuning com feature extraction e classificador customizado
3. **Otimização de hiperparâmetros** — Bayesian Search com Optuna
4. **Validação cruzada** — K-Fold Cross Validation para avaliação robusta
5. **Análise de interpretabilidade** — Grad-CAM para visualização de regiões de atenção
6. **Inferência geoespacial** — Classificação tile-a-tile de ortomosaicos com geração de shapefiles
7. **Plano de pulverização** — Geração de áreas de aplicação de defensivos baseadas na classificação

---

## 🛠 Tecnologias Utilizadas

| Categoria | Tecnologias |
|---|---|
| **Deep Learning** | PyTorch, TorchVision, PyTorch Ignite |
| **Otimização** | Optuna (Bayesian Search) |
| **Geoprocessamento** | Rasterio, GeoPandas|
| **Processamento de Imagem** | OpenCV, Pillow, scikit-image |
| **Ciência de Dados** | NumPy, scikit-learn, Matplotlib, Seaborn |

---
---

## Visão Geral

O projeto utiliza **transfer learning** com 4 arquiteturas CNN pré-treinadas no dataset PlantVillage para classificar folhas de soja em duas categorias: **saudável** e **doente**. Os modelos são então aplicados em ortomosaicos georreferenciados obtidos por drone para gerar mapas de classificação e planos de pulverização.

### Modelos Utilizados

| Modelo | Pesos Base (PlantVillage) | Melhor Val Accuracy (Bayesian Search) |
|---|---|---|
| **ResNet-101** | `ResNet_101_ImageNet_plant-model-84.pth` | 98.53% |
| **DenseNet-201** | `densenet201-model-95.pth` | 93.75% |
| **VGG-19** | `VggNet19-model-96.pth` | 97.80% |
| **MobileNetV3-Large** | `mobilenet_v3_large-model-84.pth` | 98.90% |

----
Modelos obtidos em: https://pd.dd.samlab.cn/download.html

## Fluxo de Trabalho

### 1. Preparar o Dataset
```bash
# Dividir dataset em train/val/test (75/15/10)
datasets-scripts/automatizate_split_dataset.py

# Redimensionar para 224x224
datasets-scripts/resize_images.py
```

### 2. Busca de Hiperparametros
Notebooks em `hyperparameter-search/` para encontrar os melhores hiperparametros via Optuna.

### 3. Treinar os Modelos
Notebooks em `training/` -- eles ja utilizam os hiperparametros ótimos encontrados durante a busca baesiana.

### 4. Validacao Cruzada
Notebooks em `kfold-validation/` para K-Fold Cross Validation.

### 5. Analise de Interpretabilidade
Notebook em `analysis/grad_cam_heatmaps.ipynb` para gerar mapas de calor Grad-CAM.

### 6. Inferencia em Ortomosaico
```bash
# Inferencia basica
ortho-inference/orthomosaic_inference.py --model-name densenet201

# Com filtro NDVI
ortho-inference/orthomosaic_ndvi_filter.py

# Gerar plano de pulverizacao
ortho-inference/spray_plan_generator.py
```

---
Este projeto é de uso academico.