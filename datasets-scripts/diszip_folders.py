# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:13:10 2024

@author: camila

Código destinado a deszipar pastas.
"""

import os
import zipfile
import shutil
from dotenv import load_dotenv

load_dotenv()


def extract_zip_files(zip_path, destiny_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destiny_path)


final_path = os.getenv('ZIP_FOLDER')
zip_folder = os.getenv('DISZIP_FOLDER')


for folder in os.listdir(zip_folder):
    path_zip = os.path.join(zip_folder, folder)
    # Criação da nova pasta com o nome da pasta original
    name_extract_folder = os.path.splitext(folder)[0]
    final_destiny = os.path.join(final_path, name_extract_folder)
    extract_zip_files(path_zip, final_destiny)
