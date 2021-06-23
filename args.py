# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 22:45:01 2021
@author: JerryDai

Revised on Sun Jun 13 20:44:52 2021
@author: Jay Liao
"""
import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='Deep Learning - Final Project: Traditional Chinese Handwriting Recognition')

    # Data
    parser.add_argument('-Pdt', '--dtPATH', type=str, default='./data/')
    parser.add_argument('-Pt', '--trainTarget', type=str, default='8')
    parser.add_argument('-Pms', '--msPATH', type=str, default='./model/')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.9)
    parser.add_argument('-mn', '--model_name', type=str, default = 'class8')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='No. of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-d', '--DEVICE', type=str, default='cuda', choices=['cuda', 'cuda:0', 'cuda:1', 'cpu'])
    parser.add_argument('-mt', '--model_type', type=str, default='resnet')

    return parser
