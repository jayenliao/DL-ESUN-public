# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 22:47:52 2021
@author: JerryDai

Revised on Sun Jun 13 20:44:52 2021
@author: Jay Liao
"""
from args import get_args
from trainer import Trainer

def main(args):
    
    trainer = Trainer(
        dataPATH = args.dtPATH,
        trainTarget = args.trainTarget,
        modelsavePATH = args.msPATH,
        checkpoint = args.checkpoint,
        learning_rate = args.lr,
        val_ratio = args.val_ratio,
        model_name = args.model_name,
        epochs = args.epochs,
        batch_size = args.batch_size,
        DEVICE = args.DEVICE,
        model_type = args.model_type
    )
    
    trainer.train()
    trainer.plot_result()
    
if __name__ == '__main__':
    args = get_args().parse_args()
    print(args.epochs)
    main(args)
