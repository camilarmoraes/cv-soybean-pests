# -*- coding: utf-8 -*-
"""
Grad-CAM aplicado em tiles individuais do ortomosaico.

Gera mapas de calor (heatmaps) usando Grad-CAM para visualizar quais
regiões de cada tile o modelo CNN está focando para tomar a decisão
de classificação.

# NOTA: O melhor modelo para análise foi o DenseNet-201,
# que obteve a melhor acurácia geral nos experimentos do TCC.

Baseado em: ORTOMOSAIC/codes/10_mapa_de_calor.py

Uso:
    $ python tile_heatmap.py --input-dir ./tiles --model-path model.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from orthomosaic_inference import load_model

load_dotenv()

# ============================================================
# CONFIGURAÇÃO
# ============================================================
MODEL_PATH = os.getenv("TRAINED_MODEL_PATH", "./models/best_model.pt")
MODEL_NAME = os.getenv("MODEL_NAME", "densenet201")
OUTPUT_DIR = os.getenv("ORTHOMOSAIC_OUTPUT_DIR", "./results/heatmaps")


class GradCAM:
    """
    Implementação do Grad-CAM (Gradient-weighted Class Activation Mapping).

    Utiliza hooks na última camada convolucional do modelo para capturar
    as ativações e gradientes, gerando um mapa de calor que indica as
    regiões mais relevantes para a decisão do classificador.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Gera o mapa de calor Grad-CAM para a imagem de entrada.

        Args:
            input_tensor: Tensor (1, C, H, W) normalizado.
            target_class: Classe alvo. Se None, usa a classe predita.

        Returns:
            Mapa de calor (numpy array H x W) normalizado entre 0 e 1.
        """
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0)

        # Normalizar
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def get_target_layer(model, model_name):
    """Retorna a última camada convolucional do modelo para Grad-CAM."""
    if model_name == "resnet101":
        return model.layer4[-1].conv3
    elif model_name == "densenet201":
        return model.features.denseblock4.denselayer32.conv2
    elif model_name == "vgg19":
        return model.features[-1]
    elif model_name == "mobilenetv3":
        return model.features[-1]
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")


def generate_heatmap(image_path, model, grad_cam, device, output_path=None):
    """
    Gera e salva o mapa de calor Grad-CAM para uma imagem.

    Args:
        image_path: Caminho da imagem de entrada.
        model: Modelo CNN.
        grad_cam: Instância de GradCAM.
        device: Dispositivo (cuda/cpu).
        output_path: Caminho para salvar o heatmap.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = grad_cam.generate(input_tensor)

    # Redimensionar CAM para o tamanho da imagem
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))

    # Plotar resultado
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image.resize((224, 224)))
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Sobreposição")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap salvo em: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM em tiles individuais")
    parser.add_argument("--input-dir", required=True, help="Diretório com tiles para análise")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_name, args.model_path, device)
    target_layer = get_target_layer(model, args.model_name)
    grad_cam = GradCAM(model, target_layer)

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"heatmap_{filename}")
            generate_heatmap(input_path, model, grad_cam, device, output_path)

    print("Geração de heatmaps finalizada!")


if __name__ == "__main__":
    main()
