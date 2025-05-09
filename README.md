# Detecção de Pragas em Plantas de Soja com Visão Computacional

Este repositório contém os códigos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) intitulado:  
**"Avaliação da Eficácia de Detecção de Pragas em Culturas de Soja Utilizando Visão Computacional"**.

## 📌 Contexto do Projeto

O trabalho tem como foco a avaliação de modelos de redes neurais convolucionais pré-treinados para a detecção de doenças em plantas de soja.

O processo de treinamento foi realizado com imagens provenientes de bases de dados públicas, nas quais as plantas foram fotografadas em ambientes terrestres (com proximidade a planta). A partir desses modelos treinados, buscou-se avaliar o desempenho deles em um novo cenário: **a inferência sobre imagens capturadas por drones em ambiente aéreo**, considerando fatores como **altura de voo, angulação da câmera e velocidade da aeronave**.

## 🧪 Objetivos

- Integrar e preparar conjuntos de dados com imagens de plantas de soja;
- Realizar o pré-processamento adequado para o treinamento dos modelos;
- Aplicar _fine-tuning_ nos modelos convolucionais pré-treinados selecionados;
- Avaliar a eficácia dos modelos em imagens aéreas;
- Aplicar técnicas de geoprocessamento para mapear e criar um plano de pulverização.

## 🛠 Tecnologias Utilizadas

- Python (3.10.10)
- OpenCV
- GDAL / Rasterio
- Scikit-learn
- Pytorch (Torchvision, Optuna, Pytorch-Ignite)
- QGIS (Software de Sistema de Informação Geográfica)

---
