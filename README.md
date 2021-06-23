# Deep Learning - 玉山人工智慧挑戰賽2021夏季賽：中文手寫影像辨識

本組參加「TBrain AI實戰吧」競賽平上的「玉山人工智慧挑戰賽2021夏季賽」，進行中文手寫影像辨識的任務，並將訓練好的模型部署在RESTful API Server上。比賽目標是預測包含一個文字的影像資料判別，目標類別共801類，分別為800字與1類isnull，故目標任務為Long Tail Classification，其中挑戰包含：繁體中文手寫字辨識資源較少、無部署API經驗以及資料包含雜訊，並且比賽方為模擬真實情況ground true label有許多錯誤需要清理。我們將在訓練模型前，對資料進行若干個前處理，包含：人工清理ground true label、Data augmentation（填白、resize、旋轉角度、水平翻轉...等)，接著為解決Long Tail 問題，將800字分為12群，以階層式多model進行訓練，將外部模型的confidence output當作權重乘上內部模型預測的topk取argmax，獲得最終預測結果，骨幹網路嘗試Resnet-50和EfficientNet-b4，實驗結果EfficientNet-b4較佳且較穩定，在驗證集中內外模型都有約9成以上的Macro F1，但在測試集中效果不佳，推測原因為比賽方資料較多雜訊，而訓練集中我們為放入這類資料導致模型無法辨識這類影像。

## Data

### Images

Due to rules of the competition, the following data can not be accessed currently.

- Raw competition data: `train.zip`, containing 68,804 images

- Pretrained data: `pretrain.zip`, containing 250,712 images

- `./output_2021-06-18/` contains testing images and prediction data of formal cp 4 on 18 June

    - `~/cleaned_request/` contains labeled testing images requested on 18 June.

- `./output_2021-06-17/` contains testing images and answer data of formal cp 3 on 17 June

- `./output_2021-06-08_cleaned/` contains testing images, data, and answer data of testing cp 4 on 8 June

- `./data/`:

    - `./data/Class12` contains cleaned and re-labeled images for the exterior model

    - Other subfolders under `./data/` contains cleaned and re-labeled images with clustering for 12 interior models (0-11).

- \[IMPORTANT\] `./TestClean/`: the latest image data cleaned on 14-15 June. Please use this data foloder for future training and prediction.

### Word dictionary

- `training_data_dic_800.txt`: the word dictionary of competition data with 800 labels

- `training_data_dic_4803.txt`: the word dictionary of pretrained data with 4,803 labels

## Models

- `./model/`: Means, standard deviations, and model outputs (checkpoints, result plots).

    -  `./model/pretrained/` contains the pretrained model means,stds, and checkpoint with 801 words.

    - `./model/Class0/`: model checkpoints of interior model 0
        - `~/15-12-13-31/`: ResNet
        - `~/19-19-40-02/`: EfficientNet

    - `./model/Class1/`: model checkpoints of interior model 1.
        - `~/15-12-12-44/`: ResNet
        - `~/19-19-38-34/`: EfficientNet
    
    - ... and so on until `./model/Class11`
        - `~/15-hh-mm-ss/`: ResNet
        - `~/19-hh-mm-ss/`: EfficientNet

## Code and Usage

### Model training and evaluation

- `args.py` defines the parser of arguments.
- `main.py`: the main program of training and evaluation.
- `trainer.py` define the trainer for training models with evaluation.
- `model.py` define the model structures.
- `predict.py` define the process of prediction with model structures in model.py.
- `predict_test.py` the example of predict.py.

### Others

- `./plots/`: training curves of all models.

- `EDA.ipynb`: Exploratory Data Analysis on the raw data with some simple descriptive statistics.

- `dir_maker.py`: create empty subfolders with labels in `word_dict.csv`.

- `resnet_onOCR.py`: train ResNet50.

- `predict_image.py`: load means, standard deviations, and model weights for training to predict new image inductively.

- `api.py`: code for activating API, including the prediction function.
