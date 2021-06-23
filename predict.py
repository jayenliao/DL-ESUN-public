import os
import numpy as np
import pandas as pd
import model as m # import ResNetConfig, Bottleneck, ResNet
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from collections import namedtuple
from efficientnet_pytorch import EfficientNet

class Predictor():
    def __init__(self, modelPATH:str, word_dict_txt:str, mapping_df:str, model_type:str):
        self.modelPATH = modelPATH
        self.word_dict = self.load_word_dict(word_dict_txt)
        self.mapping_df = self.read_pkl(mapping_df)
        self.model_table = pd.DataFrame([['class0', 66],
                                         ['class1', 91],
                                         ['class2', 63],
                                         ['class3', 76],
                                         ['class4', 54],
                                         ['class5', 75],
                                         ['class6', 37],
                                         ['class7', 90],
                                         ['class8', 100],
                                         ['class9', 37],
                                         ['class10', 59],
                                         ['class11', 63],
                                         ], columns = ['model_name', 'class_number'])
        
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        self.resnet50_config = ResNetConfig(block = m.Bottleneck,
                                            n_blocks = [3, 4, 6, 3],
                                            channels = [64, 128, 256, 512])
        
        # modelPATH = './model/'
        self.AllClass_model_folder = os.path.join(modelPATH, 'AllClass')
        self.AllClass_model = self.load_model(self.AllClass_model_folder, class_number = 12) # class_number = 12
        
    
    def load_word_dict(self, word_dict_txt):
        dict801 = list(pd.read_table(word_dict_txt, sep=',')['word_dict'])
        word_dict = {}
        for i, w in enumerate(dict801):
            word_dict[i] = w
        
        return word_dict
    
    
    def load_model(self, model_folder, class_number, model_type):
        # print(model_pt)
        if 'res' in model_type.lower():
            model = m.ResNet(self.resnet50_config, class_number)
        elif 'eff' in model_type.lower() or 'b4' in model_type.lower():
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=class_number)
        
        model = m.ResNet(self.resnet50_config, class_number)
        model = model.to(self.device)
        lst = os.listdir(model_folder)
        lst.sort()
        dt = lst[-1]
        if 'AllClass' in model_folder:
            pt = os.path.join(model_folder, 'checkpoint', 'checkpoint.pt')
        else:
            pt = os.path.join(model_folder, dt, 'checkpoint', 'checkpoint.pt')

        try:
            model_state = torch.load(pt, map_location=torch.device('cpu'))
        except:
            print(pt)
            raise
        # print(model_state)
        model.load_state_dict(model_state)
        
        ## load pretrained parameters
        if 'res' in model_type.lower():
            if 'AllClass' in model_folder:
                model.load_pretrained_parameter(os.path.join(model_folder, 'pretrained'))
            else:
                model.load_pretrained_parameter(os.path.join(model_folder, dt, 'pretrained'))

        return model
    
    def predict_single(self, model, image, model_type, k):
        
        model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            
            if 'res' in model_type.lower():
                y_pred, _ = model(image)
                
            elif 'eff' in model_type.lower() or 'b4' in model_type.lower():
                y_pred = model(image)
                
            # y_pred, _ = model(image)
            y_prob = F.softmax(y_pred, dim = -1)
            # top_pred = y_prob.argmax(1, keepdim = True)
            top_prob = y_prob.topk(k, 1)

        return top_prob
    
    def predict(self, image, outer_model_type, inner_model_type, topk=5):
        """ Predict your model result.
    
        @param:
            image (numpy.ndarray): an image.
        @returns:
            prediction (str): a word.
        """
    
        prediction = pd.DataFrame([], columns = ['confidence', 'inner_pred'])

        outer_probs = self.predict_single(self.AllClass_model, image, model_type=outer_model_type, k=int(topk)) # get topk labels
        outer_conf, outer_call = outer_probs
        outer_conf = outer_conf.cpu().numpy()[0]
        outer_call = outer_call.cpu().numpy()[0]
        for conf, call in zip(outer_conf, outer_call):
            model_name = self.model_table.loc[call]['model_name']
            class_number = self.model_table.loc[call]['class_number']
            # print(os.path.join(self.modelPATH, model_name))
            try:
                tmp_model = self.load_model(os.path.join(self.modelPATH, model_name), class_number)
            except:
                print('\nmodel_name', model_name)
                raise
            inner_prob = self.predict_single(tmp_model, image, model_type=inner_model_type, k=1) # get top1
            inner_call = int(inner_prob[1].cpu().numpy())
            search_bool_index = np.logical_and(self.mapping_df['outer_index'] == call,
                                               self.mapping_df['inner_index'] == inner_call)
        
            search_result = self.mapping_df[search_bool_index]
        
            confidence = float(inner_prob[0].cpu().numpy()[0]) * conf
            inner_pred = search_result['word'].values[0]
        
            prediction = prediction.append(pd.DataFrame([[confidence, inner_pred]], columns = ['confidence', 'inner_pred']))

        # outer_probs = self.predict_single(self.AllClass_model, image, k = topk) # get top5
        # outer_call = outer_probs[1].cpu().numpy()[0]
        # # print(outer_call)
        # for call in outer_call:
        #     model_name = self.model_table.loc[call]['model_name']
        #     class_number = self.model_table.loc[call]['class_number']
        #     try:
        #         tmp_model = self.load_model(os.path.join(self.modelPATH, model_name), class_number)
        #     except:
        #         print('\nmodel_name', model_name)
        #         raise
        #     inner_prob = self.predict_single(tmp_model, image, k = 1) # get top1
        #     inner_call = int(inner_prob[1].cpu().numpy())
        #     search_bool_index = np.logical_and(self.mapping_df['outer_index'] == call,
        #                                        self.mapping_df['inner_index'] == inner_call)
        
        #     search_result = self.mapping_df[search_bool_index]
        
        #     confidence = float(inner_prob[0].cpu().numpy()[0])
        #     inner_pred = search_result['word'].values[0]
        
        #     prediction = prediction.append(pd.DataFrame([[confidence, inner_pred]], columns = ['confidence', 'inner_pred']))
        
        prediction = prediction.reset_index(drop = True)
        pred = prediction.loc[prediction['confidence'].argmax(), 'inner_pred']
        
        if _check_datatype_to_string(pred):
            return pred, prediction
    
    def read_pkl(self, pkl_file_name):
        f = open(pkl_file_name, 'rb')
        pkl_data = pickle.load(f)
        
        return pkl_data

def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')