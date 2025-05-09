# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:15:27 2024

@author: Camila

Código para automatizar o split do dataset.
O diretório raiz possui subpastas com as imagens.
"""

import os
from dotenv import load_dotenv
import shutil
from sklearn.model_selection import train_test_split

load_dotenv()

# localização do pasta original com as imagens e a pasta final alvo
root_dir = os.getenv('ROOT_DIR_SPLIT')
target_dir = os.getenv('TARGET_DIR_SPLIT')


def create_target_dirs(target_dir):
    train_healthy = os.path.join(os.path.join(target_dir, "train"), "healthy")
    train_diseases = os.path.join(os.path.join(target_dir, "train"), "diseases")

    val_healthy = os.path.join(os.path.join(target_dir, "val"), "healthy")
    val_diseases = os.path.join(os.path.join(target_dir, "val"), "diseases")

    test_healthy = os.path.join(os.path.join(target_dir, "test"), "healthy")
    test_diseases = os.path.join(os.path.join(target_dir, "test"), "diseases")

    for directory in [
        train_healthy, train_diseases, val_healthy,
        val_diseases, test_healthy, test_diseases
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def split_dataset(files, train_percent=0.75, test_percent=0.10):
    # val_percent = 0.15
    # Código que automatiza o split dos dados, com base em uma porcentagem dada
    t_size = 1 - train_percent
    x_size = test_percent / t_size

    train_files, temp_files = train_test_split(files, test_size=t_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=x_size, random_state=42)

    return train_files, val_files, test_files


def copy_dataset(root_dir, target_dir):
    first_dirs = os.listdir(root_dir)
    for dirs in first_dirs:
        # diretórios dentro de um dataset inteiro
        internals_dir = os.listdir(os.path.join(root_dir, dirs))
        for int_dirs in internals_dir:
            files = os.listdir(os.path.join(os.path.join(root_dir, dirs), int_dirs))
            if int_dirs == 'Healthy':
                final_dir = 'healthy'
            else:
                final_dir = 'diseases'
            train_files, val_files, test_files = split_dataset(files)

            train_folder = os.path.join(os.path.join(target_dir, "train"), final_dir)
            val_folder = os.path.join(os.path.join(target_dir, "val"), final_dir)
            test_folder = os.path.join(os.path.join(target_dir, "test"), final_dir)

            for file in train_files:
                try:
                    shutil.copy(os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file),
                            os.path.join(train_folder, file)
                            )
                except Exception as e:
                    print(f"Erro ao copiar {os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file)} para {os.path.join(train_folder, file)}: {e}")

            for file in val_files:
                try:

                    shutil.copy(os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file),
                            os.path.join(val_folder, file)
                            )
                except Exception as e:
                    print(f"Erro ao copiar {os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file)} para {os.path.join(train_folder, file)}: {e}")

            for file in test_files:
                try:

                    shutil.copy(os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file),
                            os.path.join(test_folder, file))
                except Exception as e:
                    print(f"Erro ao copiar {os.path.join(os.path.join(os.path.join(root_dir, dirs), int_dirs), file)} para {os.path.join(train_folder, file)}: {e}")


create_target_dirs(target_dir)
copy_dataset(root_dir, target_dir)
