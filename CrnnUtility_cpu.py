# -*- coding: utf-8 -*

import os
import fnmatch
import cv2
import numpy as np
import string
import time
from PIL import Image
import shutil
import requests
import logging
import json
import sys

from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model 
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import datetime
from numba import cuda
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.backend import get_session
import tensorflow
import gc

extension = '.hdf5'
api_UpdateModel = 'https://graphicsapi.quanta-camp.com/api/model/update'
# 統一訓練資料路徑 ＆ 訓練檔案的路徑
# 本機測試(Test)
# filepath = '/Users/ccasd/Desktop/'
# output = '/Users/ccasd/Desktop/QRDC_All_PlateNumber'
# hdf5_path = '/Users/ccasd/Desktop/'
# Docker配合NFS(Production)
filepath = '/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/uploads'  # 訓練檔案存放路徑，資料夾名稱由API提供(+FileName)
output = '/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/CRNNData'  # 固定存放訓練資料的位置
hdf5_path = '/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/crnn/model'  # 訓練結束後hdf5存放路徑(+ModelName.hdf5)
tmp_path = '/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/crnn/model'
#hdf5_path_test = '/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/crnn/model'
hdf5_path_test = './testfolder'
retry_time = 0
list_filename = []  # 存放訓練後的檔案名稱
list_threshold = []  # 存放訓練後的準確率

ModelName = sys.argv[1]
TestFilePath = sys.argv[2].split(',')
CreateUser = sys.argv[3]
SetValue = sys.argv[4]
#print('SetValue>>>' + sys.argv[4])
loggings = sys.argv[5]
"""測試"""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# parse parameter
#now = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, filename=loggings, filemode='a', format=FORMAT)
logging.info(SetValue)
#jsonval = json.loads(SetValue)
char_list = SetValue#str(jsonval['char_list'])
#char_list = SetValue['char_list']

if len(TestFilePath) == 0:
    print("TestFilePath list not null.")
else:
    try:
        list_Result = []
        valid_img = []


        #for x in TestFilePath:
            #logging.info(os.path.dirname(x))
            #for root, dirnames, filenames in os.walk(os.path.join(cls.filepath, x, CreateUser)):
        #for root, dirnames, filenames in os.walk(os.path.dirname(TestFilePath[0])):
        #    #for f_name in fnmatch.filter(filenames, '*.jpg'):
        #    for f_name in filenames:
        #        if f_name.lower().endswith(('.png','.jpg','jpeg')):
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread('./' + str(sys.argv[2])), cv2.COLOR_BGR2GRAY)
        
        #print('./' + str(sys.argv[2]))
        # convert each image of shape (32, 128, 1)
        w, h = img.shape

        # convert to  32*128
        img = cv2.resize(img, (128, 32), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=2)
        # Normalize each image
        img = img / 255.

        valid_img.append(img)
        
        
        valid_img = np.array(valid_img)
        inputs = Input(shape=(32, 128, 1))

        # convolution layer with kernel size (3,3)
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        # poolig layer with kernel size (2,2)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        # poolig layer with kernel size (2,1)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        # Batch normalization layer
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
        
        # bidirectional LSTM layers with units=128
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

        outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)
        #fullfilepath = os.path.join(cls.hdf5_path, ModelName, ModelName + cls.extension)
        fullfilepath = os.path.join(hdf5_path_test, ModelName + extension)
        act_model = Model(inputs, outputs)
        

        #fullfilepath = os.path.join(hdf5_path_test, ModelName, ModelName + extension)
        #act_model = load_model(fullfilepath,compile=False)
        #act_model = load_model(fullfilepath)
        act_model.load_weights(fullfilepath)

        #act_model.summary()

        logging.info(fullfilepath)
        logging.info(str(len(valid_img)))
        # predict outputs on validation images
        prediction = act_model.predict(valid_img[:len(valid_img)])
        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],greedy=True)[0][0])

        # see the results
        for x in out:
            bbb = ''
            for p in x:
                if int(p) != -1:
                    bbb = bbb + char_list[int(p)]

            list_Result.append(bbb)

        #K.clear_session()
        #cuda.select_device(0)
        #cuda.close()
    except Exception as e:
        print(e)
    print(list_Result)


def callApi(cls, url, jsonData):
    proxies = {
        "http": "http://10.243.17.28:80",
        "https": "http://10.243.17.28:80",
    }
    logging.info('****************************************************************************************************************************')
    logging.info('URL : ' + url)
    logging.info('Body : ' + json.dumps(jsonData))
    # sending post request and saving response as response object
    r = requests.post(url=url, json=jsonData, headers={"Connection": "close"}, verify=False, timeout=60, proxies=proxies)
    logging.info('StatusCode : ' + str(r.status_code))
    logging.info('Response : ' + r.text)
    logging.info('****************************************************************************************************************************')
    return r

