# ESUN-OCR-Chinese

## Data

### Images

Please go [here](https://drive.google.com/drive/folders/180dmucYW9MXQccDQ0QPqS6ct9gJPZI1A?usp=sharing) to access the following data.

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

- `EDA.ipynb`: Exploratory Data Analysis on the raw data with some simple descriptive statistics.

- `dir_maker.py`: create empty subfolders with labels in `word_dict.csv`.

- `resnet_onOCR.py`: train ResNet50.

- `predict_image.py`: load means, standard deviations, and model weights for training to predict new image inductively.

- `api.py`: code for activating API, including the prediction function.
