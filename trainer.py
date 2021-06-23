"""
Created on Sat Jun 12 03:53:02 2021
@author: JerryDai

Modified by Jay Liao on Tue Jun 15
"""
import random, os, copy, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model as m # import ResNetConfig, Bottleneck, ResNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data

from tqdm import tqdm
from PIL import Image
from collections import namedtuple
from datetime import datetime
from efficientnet_pytorch import EfficientNet

cudnn.benchmark = True

class Trainer: 
    def __init__(self, dataPATH:str, trainTarget:str, modelsavePATH:str,
                 checkpoint:str, learning_rate:float, val_ratio:float,
                 model_name:str, epochs:int, batch_size:int, DEVICE:str, model_type:str):
        
        print('Model Initialing Phase')
        
        self.dataPATH = dataPATH
        self.trainTarget = trainTarget
        train_data_folder = os.path.join(self.dataPATH, self.trainTarget)
        dt = datetime.now().strftime('%d-%H-%M-%S') 
        self.modelsavePATH = os.path.join(modelsavePATH, model_name, dt)
        self.model_name = model_name
        self.checkpoint_folder = os.path.join(self.modelsavePATH, 'checkpoint')
        self.pretrained_folder = os.path.join(self.modelsavePATH, 'pretrained')
        self.pretrain_modelPATH = os.path.join(self.pretrained_folder, 'pretrained_model.pt')
        self.result_folder = os.path.join(self.modelsavePATH, 'result')
        self.checkpoint = checkpoint
        self.learning_rate = learning_rate
        self.val_ratio = val_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.pretrained_size = 32
        self.model_type = model_type
        
        seed = 9101
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        ## Judge folder is empty
        if os.path.exists(self.modelsavePATH):
            if not os.path.exists(self.checkpoint_folder):
                os.makedirs(self.checkpoint_folder)
                
            if not os.path.exists(self.pretrained_folder):
                os.makedirs(self.pretrained_folder)
                calculate_pretrain = True
                
            if not os.path.exists(self.result_folder):
                os.makedirs(self.result_folder)
        else:
            os.makedirs(self.modelsavePATH)
            os.makedirs(self.checkpoint_folder)
            os.makedirs(self.pretrained_folder)
            os.makedirs(self.result_folder)
        
        ## Calculate pretrained parameters
        pretrain_means_path = os.path.join(self.pretrained_folder,'pretrained_means.pt')
        pretrain_stds_path = os.path.join(self.pretrained_folder,'pretrained_stds.pt')
        calculate_pretrain = not np.logical_and(os.path.exists(pretrain_means_path),
                                                os.path.exists(pretrain_stds_path))
        
        if calculate_pretrain:
            train_data = datasets.ImageFolder(
                root = train_data_folder, transform = transforms.ToTensor()
            )
            means = torch.zeros(3)
            stds = torch.zeros(3)
            for img, label in tqdm(train_data):
                means += torch.mean(img, dim = (1,2))
                stds += torch.std(img, dim = (1,2))
            means /= len(train_data)
            stds /= len(train_data)
            pretrained_means = means.cpu().numpy() # [0.485, 0.456, 0.406]
            torch.save(pretrained_means, os.path.join(self.pretrained_folder,'pretrained_means.pt'))
            pretrained_stds= stds.cpu().numpy() # [0.229, 0.224, 0.225]
            torch.save(pretrained_stds, os.path.join(self.pretrained_folder,'pretrained_stds.pt'))
        
        else:
            pretrained_means = torch.load(os.path.join(self.pretrained_folder, 'pretrained_means.pt'))
            pretrained_stds = torch.load(os.path.join(self.pretrained_folder, 'pretrained_stds.pt'))

        ## Set up data transform
        train_transforms = transforms.Compose([
                                    lambda x: fill_blank(x),
                                    transforms.Resize(self.pretrained_size),
                                    transforms.RandomRotation(3),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomCrop(self.pretrained_size, padding = 10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = pretrained_means, 
                                                         std = pretrained_stds)
                                    ])
        
        test_transforms = transforms.Compose([
                                    lambda x: fill_blank(x),
                                    transforms.Resize(self.pretrained_size),
                                    transforms.CenterCrop(self.pretrained_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = pretrained_means, 
                                                         std = pretrained_stds)
                                    ])
        
        ## Load training data
        train_data = datasets.ImageFolder(root = train_data_folder, 
                                          transform = train_transforms)
        
        n_train_examples = int(len(train_data) * self.val_ratio)
        n_valid_examples = len(train_data) - n_train_examples
        
        train_data, valid_data = data.random_split(train_data, 
                                                   [n_train_examples, n_valid_examples])
        
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms
        
        self.train_iterator = data.DataLoader(train_data, 
                                              shuffle = True, 
                                              batch_size = self.batch_size)
        
        self.valid_iterator = data.DataLoader(valid_data, 
                                              batch_size = self.batch_size)
        
        ## Set up model Structure
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet50_config = ResNetConfig(block = m.Bottleneck,
                                       n_blocks = [3, 4, 6, 3],
                                       channels = [64, 128, 256, 512])
        
        if self.model_type == 'resnet':
            ## Load model weights or not
            if checkpoint is None:
                pretrained_model = models.resnet50(pretrained=True)
                IN_FEATURES = pretrained_model.fc.in_features 
                OUTPUT_DIM = 801
                fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
                pretrained_model.fc = fc
                pretrained_mode_dict = torch.load(os.path.join(modelsavePATH, 'pretrained', 'pretrain_model.pt'))
                pretrained_model.load_state_dict(pretrained_mode_dict)
    
                ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
                resnet50_config = ResNetConfig(
                    block = m.Bottleneck,
                    n_blocks = [3, 4, 6, 3],
                    channels = [64, 128, 256, 512]
                )
    
                IN_FEATURES = pretrained_model.fc.in_features
                OUTPUT_DIM = len(train_data.dataset.classes)
                fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
                pretrained_model.fc = fc
                self.model = m.ResNet(resnet50_config, OUTPUT_DIM)
                self.model.load_state_dict(pretrained_model.state_dict())
        
            else:
                OUTPUT_DIM = len(train_data.dataset.classes)
                self.model = m.ResNet(resnet50_config, OUTPUT_DIM)
                self.model.load_state_dict(torch.load(self.checkpoint))
        elif self.model_type == 'efficientnet-b4':
            if checkpoint is None:
                OUTPUT_DIM = len(train_data.dataset.classes)
                self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=OUTPUT_DIM)
            else:
                OUTPUT_DIM = len(train_data.dataset.classes)
                self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=OUTPUT_DIM)
                self.model.load_state_dict(torch.load(self.checkpoint))
        
        if self.model_type == 'resnet':
            ## Set up parameters of model
            params = [
              {'params': self.model.conv1.parameters(), 'lr': self.learning_rate / 10},
              {'params': self.model.bn1.parameters(), 'lr': self.learning_rate / 10},
              {'params': self.model.layer1.parameters(), 'lr': self.learning_rate / 8},
              {'params': self.model.layer2.parameters(), 'lr': self.learning_rate / 6},
              {'params': self.model.layer3.parameters(), 'lr': self.learning_rate / 4},
              {'params': self.model.layer4.parameters(), 'lr': self.learning_rate / 2},
              {'params': self.model.fc.parameters()}
             ]
            self.optimizer = optim.Adam(params, lr = self.learning_rate)
        
        elif self.model_type == 'efficientnet-b4':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        STEPS_PER_EPOCH = len(self.train_iterator)
        TOTAL_STEPS = (self.epochs + 1) * (STEPS_PER_EPOCH)
        
        MAX_LRS = [p['lr'] for p in self.optimizer.param_groups]
        
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer,
                                                 max_lr = MAX_LRS,
                                                 total_steps = TOTAL_STEPS)

    def train(self):
        #self.best_valid_loss = float('inf')
        self.best_valid_macro_F1 = 0

        self.epoch_train_loss = []
        self.epoch_valid_loss = []
        
        self.epoch_train_acc_1 = []
        self.epoch_valid_acc_1 = []
        
        self.epoch_train_acc_5 = []
        self.epoch_valid_acc_5 = []
        
        self.epoch_train_macro_percision = []
        self.epoch_train_macro_recall = []
        self.epoch_train_macro_F1 = []
        
        self.epoch_valid_macro_percision = []
        self.epoch_valid_macro_recall = []
        self.epoch_valid_macro_F1 = []
        
        print('Model Training Phase')
        dtt = datetime.now().strftime('%y%m%d_%H%M%S')
        for epoch in range(self.epochs):
            start_time = time.monotonic()
            train_loss, train_acc_1, train_acc_5, train_precision, train_recall = self.train_single(self.model, self.train_iterator, self.optimizer, self.criterion, self.scheduler, self.device)
            valid_loss, valid_acc_1, valid_acc_5, valid_precision, valid_recall = self.evaluate(self.model, self.valid_iterator, self.criterion, self.device)
            
            train_macro_percision = float(np.mean(train_precision, axis = 1))
            train_macro_recall = float(np.mean(train_recall, axis = 1))
            train_macro_F1 = float(np.mean(2 * train_precision * train_recall / (train_precision + train_recall), axis = 1))
        
            valid_macro_percision = float(np.mean(valid_precision, axis = 1))
            valid_macro_recall = float(np.mean(valid_recall, axis = 1))
            valid_macro_F1 = float(np.mean(2 * valid_precision * valid_recall / (valid_precision + valid_recall), axis = 1))
            
            self.epoch_train_loss.append(train_loss)
            self.epoch_valid_loss.append(valid_loss)
            self.epoch_train_acc_1.append(train_acc_1)
            self.epoch_valid_acc_1.append(valid_acc_1)
            self.epoch_train_acc_5.append(train_acc_5)
            self.epoch_valid_acc_5.append(valid_acc_5)
            
            self.epoch_train_macro_percision.append(train_macro_percision)
            self.epoch_train_macro_recall.append(train_macro_recall)
            self.epoch_train_macro_F1.append(train_macro_F1)
            self.epoch_valid_macro_percision.append(valid_macro_percision)
            self.epoch_valid_macro_recall.append(valid_macro_recall)
            self.epoch_valid_macro_F1.append(valid_macro_F1)
        
            if valid_macro_F1 > self.best_valid_macro_F1:
                self.best_valid_macro_F1 = valid_macro_F1
                save_checkpoint(self.model_name, epoch, dtt, self.model, self.best_valid_macro_F1, self.checkpoint_folder)
        
            end_time = time.monotonic()
        
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
                  f'Train Acc @5: {train_acc_5*100:6.2f}%')        
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
                  f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
    
    def train_single(self, model, iterator, optimizer, criterion, scheduler, device):
        classes = iterator.dataset.dataset.classes
        pred_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        intersection_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        y_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        
        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        model.train()  
        for (x, y) in iterator: # iterator = train_iterator  
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if self.model_type == 'resnet':
                y_pred, _ = model(x)
                
            elif self.model_type == 'efficientnet-b4':
                y_pred = self.model(x)
                
            loss = criterion(y_pred, y)
            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
            tmp_pred_count, tmp_intersection_count, tmp_y_count = calculate_macroF1(y_pred, y, classes)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
            
            pred_count += pd.DataFrame([list(tmp_pred_count.values())], columns = list(tmp_pred_count.keys()))
            intersection_count += pd.DataFrame([list(tmp_intersection_count.values())], columns = list(tmp_intersection_count.keys()))
            y_count += pd.DataFrame([list(tmp_y_count.values())], columns = list(tmp_y_count.keys()))
            
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)
        
        precision = intersection_count / pred_count
        recall = intersection_count / y_count
        
        return epoch_loss, epoch_acc_1, epoch_acc_5, precision , recall
    
    def evaluate(self, model, iterator, criterion, device):
        classes =  iterator.dataset.dataset.classes
        pred_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        intersection_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        y_count = pd.DataFrame([np.zeros(len(classes)).astype(int)], columns = classes)
        
        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        model.eval()
        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(device)
                y = y.to(device)
                if self.model_type == 'resnet':
                    y_pred, _ = model(x)
                    
                elif self.model_type == 'efficientnet-b4':
                    y_pred = self.model(x)
                        
                loss = criterion(y_pred, y)
                acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
                tmp_pred_count, tmp_intersection_count, tmp_y_count = calculate_macroF1(y_pred, y, classes)
                epoch_loss += loss.item()
                epoch_acc_1 += acc_1.item()
                epoch_acc_5 += acc_5.item()
                pred_count += pd.DataFrame([list(tmp_pred_count.values())], columns = list(tmp_pred_count.keys()))
                intersection_count += pd.DataFrame([list(tmp_intersection_count.values())], columns = list(tmp_intersection_count.keys()))
                y_count += pd.DataFrame([list(tmp_y_count.values())], columns = list(tmp_y_count.keys()))

        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)
        precision = intersection_count / pred_count
        recall = intersection_count / y_count
        
        return epoch_loss, epoch_acc_1, epoch_acc_5, precision , recall

    def plot_result(self):

        plt.figure()
        plt.plot(self.epoch_train_loss, label = 'training')
        plt.plot(self.epoch_valid_loss, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Training Curve')
        plt.savefig(os.path.join(self.result_folder, 'Training_Curve.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.epoch_train_acc_1, label = 'training')
        plt.plot(self.epoch_valid_acc_1, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Accuracy_Top1')
        plt.savefig(os.path.join(self.result_folder, 'Accuracy_Top1.png'))
        plt.close()

        plt.figure()
        plt.plot(self.epoch_train_acc_5, label = 'training')
        plt.plot(self.epoch_valid_acc_5, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Accuracy_Top5')
        plt.savefig(os.path.join(self.result_folder, 'Accuracy_Top5.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.epoch_train_macro_percision, label = 'training')
        plt.plot(self.epoch_valid_macro_percision, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Macro_Percision')
        plt.savefig(os.path.join(self.result_folder, 'Macro_Percision.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.epoch_train_macro_recall, label = 'training')
        plt.plot(self.epoch_valid_macro_recall, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Macro_Recall')
        plt.savefig(os.path.join(self.result_folder, 'Macro_Recall.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.epoch_train_macro_F1, label = 'training')
        plt.plot(self.epoch_valid_macro_F1, label = 'validation')
        plt.legend()
        plt.grid()
        plt.title('Macro_F1')
        plt.savefig(os.path.join(self.result_folder, 'Macro_F1.png'))
        plt.close()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_checkpoint(model_name, epoch, dt, model, F1, checkpoint_folder):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    with open(os.path.join(checkpoint_folder, 'best_result.txt'), 'w') as f:
        print({'model_name': model_name, 'epoch': epoch, 'dt': dt, 'F1': F1}, file=f)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'checkpoint.pt'))

def fill_blank(image):
    width, height = image.size
    lagrge_pic_size = max(width, height)
    small_pic_size = min(width, height)
    fill_size = int(np.floor((lagrge_pic_size - small_pic_size) / 2))
    
    image = np.asarray(image)
    if  width < height:
        blank = (255 * np.ones([lagrge_pic_size, fill_size, 3])).astype(np.uint8)
        new_image = np.concatenate((blank, image, blank), axis = 1)
    else:
        blank = (255 * np.ones([fill_size, lagrge_pic_size, 3])).astype(np.uint8)
        new_image = np.concatenate((blank, image, blank), axis = 0)
    
    image = Image.fromarray(new_image)
    
    return image

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def calculate_macroF1(y_pred, y, classes, k = 1): # classes = train_iterator.dataset.dataset.classes
    
    pred_count = {key:0 for key in classes}
    intersection_count = {key:0 for key in classes}
    y_count = {key:0 for key in classes}
    
    with torch.no_grad():
        _, top_pred = y_pred.topk(k, 1)
        pred = top_pred.t().cpu().numpy().reshape(-1)
        array_y = y.cpu().numpy()
        
    for index in range(len(y)):
        pred_count[classes[pred[index]]] += 1
        y_count[classes[array_y[index]]] += 1
        
        if pred[index] == array_y[index]:
            intersection_count[classes[array_y[index]]] += 1
        
    return pred_count, intersection_count, y_count


