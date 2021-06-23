# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:55:17 2021

@author: JerryDai
"""
import os
from os import walk
from os.path import join
import re
from tqdm import tqdm
import shutil

from joblib import Parallel, delayed

# In[] def
def copy_data_to_folder(dict_key, data_path_list, target_folder): # dict_key = A[0] data_path_list = A[1]
    
    tmp_folder_path = os.path.join(target_folder, dict_key)
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)
    
    for i_index in data_path_list:
        tmp_data_name = re.split('\\\\', data_total_path[i_index])[-1]
        tmp_data_path = os.path.join(tmp_folder_path, tmp_data_name)
        shutil.copyfile(data_total_path[i_index], tmp_data_path)
    

# In[] Load Folder Data

## get word dict
root = r'D:\NCKU\Class In NCKU\DeepLearning\Final_project\OCR\data'
target_folder = os.path.join(root, 'cleaned_data(50_50)')
imgaes_folder = os.path.join(root, 'cleaned_data(50_50)-20200420T071507Z-004/cleaned_data(50_50)')

data_total_path = []
word_dict = dict()
for root, dirs, files in walk(imgaes_folder):
    for f in tqdm(files):
        fullpath = os.path.join(root, f)
        data_total_path.append(fullpath)
        
        tmp_word = re.split('_|\.', f)[0]
        
        tmp_index = data_total_path.index(fullpath)
        if tmp_word in word_dict:
            tmp_list = list(word_dict[tmp_word])
            tmp_list.append(tmp_index)
            word_dict[tmp_word] = tmp_list 
        else:
            tmp_list = list()
            tmp_list.append(tmp_index)
            word_dict[tmp_word] = tmp_list
            
## separate dict by folder
for i_key, i_index_list in tqdm(word_dict.items()):
    
    tmp_folder_path = os.path.join(target_folder, i_key)
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)
    
    for i_index in i_index_list:
        tmp_data_name = re.split('\\\\', data_total_path[i_index])[-1]
        tmp_data_path = os.path.join(tmp_folder_path, tmp_data_name)
        shutil.copy(data_total_path[i_index], tmp_data_path)
    
# _ = Parallel(n_jobs = -1)(delayed(copy_data_to_folder)(i_key, i_index_list, target_folder) for i_key, i_index_list in tqdm(word_dict.items()))


# In[] make dir

# dirlist = os.listdir(imgaes_folder)

# for i_dir in tqdm(dirlist):
    
#     tmp_folder_path = os.path.join(target_folder, i_dir)
#     if not os.path.exists(tmp_folder_path):
#         os.makedirs(tmp_folder_path)