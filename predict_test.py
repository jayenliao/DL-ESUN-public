# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:34:20 2021

@author: JerryDai
"""
import predict as predict_f
import torch
import os
import torchvision.transforms as transforms
from PIL import Image

mapping_df = 'mapping_df.pkl'
word_dict_txt = 'training_data_dic_800.txt'
modelPATH = './model/'

pretrained_size = 32
modelPATH = 'model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_means = torch.load(os.path.join(modelPATH, '801pretrained_means.pt'), map_location=torch.device('cpu'))
pretrained_stds = torch.load(os.path.join(modelPATH, '801pretrained_stds.pt'), map_location=torch.device('cpu'))
preprocess = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

image = Image.open('img_cp_test/16218515786866913_委.jpg')
image = preprocess(image)
image = image.reshape((1, 3, 32, 32))

predictor = predict_f.Predictor(modelPATH, word_dict_txt, mapping_df)
answer = predictor.predict(image, topk = 11)
print(answer)