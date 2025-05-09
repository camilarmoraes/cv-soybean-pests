# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:11:42 2024

@author: camila
"""
# Código para mudar o nome dos arquivos por questões de conflito.

import os
from dotenv import load_dotenv

load_dotenv()

root_dir = os.getenv('DIR_NAME_CHANGE')


for i, file in enumerate(os.listdir(root_dir)):
    old_file = os.path.join(root_dir, file)
    new_file = os.path.join(root_dir, f"septoria_leaf_dataset_{i+1}.jpg")
    os.rename(old_file, new_file)
