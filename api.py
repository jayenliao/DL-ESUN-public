import datetime, time
import cv2, os, pickle, base64, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from argparse import ArgumentParser
from collections import namedtuple
from PIL import Image
from model import ResNet, BasicBlock, Bottleneck
from flask import Flask, request, jsonify
import predict as predict_f

model_type = 'res'
topk = 2

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 're6091054@gs.ncku.edu.tw'
SALT = 'ncku-ds-deep-learning-re6094028-re6091054-re6104019'
#########################################

savePATH = './output_' + datetime.datetime.now().strftime('%Y-%m-%d') + '/'
try:
    os.makedirs(savePATH)
except FileExistsError:
    pass

'''
with open('training_data_dic_4803.pkl', 'rb') as f:
    d = pickle.load(f)
'''

dict800 = list(pd.read_table('training_data_dic_800.txt', sep=',')['word_dict'])
d = {}
for i, w in enumerate(dict800):
    d[i] = w

mapping_df = 'mapping_df.pkl'
word_dict_txt = 'training_data_dic_800.txt'

pretrained_size = 32
modelPATH = 'model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_means = torch.load(os.path.join(modelPATH, 'pretrained/pretrained_means.pt'), map_location=torch.device('cpu'))
pretrained_stds = torch.load(os.path.join(modelPATH, 'pretrained/pretrained_stds.pt'), map_location=torch.device('cpu'))
preprocess = transforms.Compose([
    lambda x: fill_blank(x),
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet50_config = ResNetConfig(
    block = Bottleneck,
    n_blocks = [3, 4, 6, 3],
    channels = [64, 128, 256, 512]
)

predictor = predict_f.Predictor(modelPATH, word_dict_txt, mapping_df, model_type)
print('The predictor is loaded!')

'''
model2 = ResNet(resnet50_config, 801)
model2 = model2.to(device)
model_state = torch.load(os.path.join(modelPATH, 'tut5-model_0.9806.pt'), map_location=torch.device('cpu'))
model2.load_state_dict(model_state)
'''
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

def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    '''
    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    model2.eval()
    with torch.no_grad():
        image = image.to(device)
        y_pred, _ = model2(image)
        y_prob = F.softmax(y_pred, dim = -1)
        # top_pred = y_prob.argmax(1, keepdim = True)
        #images.append(x.cpu())
        #label = y.cpu())
        #probs.append(y_prob.cpu())

    prediction = torch.argmax(y_prob, 1).cpu().numpy()
    prediction = d[int(prediction)]
    '''
    prediction = predictor.predict(image, topk=2, model_type=model_type)

    if _check_datatype_to_string(prediction):
        print(prediction)
        return prediction


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


def saveData(data, fn:str, ts:str):
    if saveDF:
        if type(data) == dict:
            try:
                data = pd.DataFrame(data)
            except:
                data = pd.DataFrame(data, index=[0])
        data.to_csv(savePATH + fn + '_' + ts + '.csv')
    else:
        with open(savePATH + fn + '_' + ts + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


@app.route("/")
def hello():
    return "Hello! This is API of team NCKU Deep Learning."


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)
    #ts = datetime.datetime.now().strftime('%H-%M-%S-%f')
    esun_timestamp = data['esun_timestamp']  # 自行取用，可紀錄玉山呼叫的 timestamp
    ts = datetime.datetime.fromtimestamp(esun_timestamp).strftime('%H-%M-%S-%f')
    tCurrent = time.time()
    #saveData(data, 'request', ts)
    if time_cost:
        print('Time cost for saving request data:', time.time() - tCurrent)

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)
    #print(image.shape)

    tCurrent = time.time()
    cv2.imwrite(savePATH + 'request_' + ts + '.png', image)
    if time_cost:
        print('Time cost for saving request image:', time.time() - tCurrent)

    # Processing the image
    image = Image.fromarray(image)
    image = preprocess(image)

    '''
    tCurrent = time.time()
    image_save = image.cpu().numpy().reshape((image_size, image_size, 3))
    cv2.imwrite(savePATH + 'preprocessed_' + ts + '.png', image_save)
    if time_cost:
        print('Time cost for saving preprocessed image:', time.time() - tCurrent)
    '''

    image = image.reshape((1, 3, image_size, image_size))
    ts_ = str(int(datetime.datetime.now().utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts_)

    # Predicting
    tCurrent = time.time()
    try:
        print(model_type)
        predictor = predict_f.Predictor(modelPATH, word_dict_txt, mapping_df, model_type)
        answer, prediction = predictor.predict(image, topk=topk, model_type=model_type)
        print(prediction)
        print('ans:', answer)
        if answer not in dict800:
            answer = 'isnull'

    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    if time_cost:
        print('Time cost for predicting:', time.time() - tCurrent)

    answer_dict = {
        'esun_uuid': data['esun_uuid'],
        'server_uuid': server_uuid,
        'answer': answer,
        'server_timestamp': datetime.datetime.now().timestamp()
    }
    tCurrent = time.time()
    
    saveData(prediction, 'prediction', ts)
    #saveData(answer_dict, 'answer', ts)
    if time_cost:
        print('Time cost for saving answer data:', time.time() - tCurrent)

    return jsonify(answer_dict)
    print(123)


if __name__ == "__main__":
    from waitress import serve
    arg_parser = ArgumentParser(usage='Usage: python ' + __file__ + ' [--port <port>] [--help]')
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    arg_parser.add_argument('-m', '--model_type', default='res')
    arg_parser.add_argument('-k', '--topk', type=int, default=2)
    arg_parser.add_argument('-s', '--saveDF', action='store_true', default=False)
    arg_parser.add_argument('-is', '--image_size', type=int, default=32)
    arg_parser.add_argument('-tc', '--time_cost', action='store_true', default=False)
    args = arg_parser.parse_args()
    image_size = args.image_size
    time_cost = args.time_cost
    saveDF = args.saveDF
    model_type = args.model_type
    topk = args.topk

    app.run(host='0.0.0.0', debug=False, port=args.port)