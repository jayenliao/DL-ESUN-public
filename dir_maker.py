# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:16:45 2021

@author: JerryDai
"""
import os
from tqdm import tqdm
import pandas as pd

target_dir = 'testdir_data'

word_dict = pd.read_csv('word_dict.csv', encoding = 'big5')

for i_dir in tqdm(word_dict['word_dict']):
    
    tmp_folder_path = os.path.join(target_dir, i_dir)
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)