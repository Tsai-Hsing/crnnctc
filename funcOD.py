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

def mainPredict(image, path, modelName, userDict , graph, sess):
    data = {}
    data['isSuccess'] = 'true'
    data['ErrorMsg'] = ''
    data['result'] = []
    try:
        list_Result = []
        valid_img = []
        image.save(path + '/original.' + userDict["FileExtension"])#image.format.lower())
        img = cv2.cvtColor(cv2.imread(path + '/original.' + userDict['FileExtension']), cv2.COLOR_BGR2GRAY)
        
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
        act_model = userDict['act_model']
        char_list = userDict['char_list']
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
        print(list_Result)
        returnarr = []
        for str in list_Result:
            tmparr = str.replace("['",'').replace("']",'').split(' ')
            tmpval = {
                'score':'',
                'BoundingBox':[]
                'Value':[tmparr[0]]
            }
            returnarr.append(tmpval)
        data['result'] = returnarr
        print(json.dumps(data))
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0] #取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        data['isSuccess']= 'false'
        data['ErrorMsg']= str(errorMsg)
    finally:
        return json.dumps(data)

def mainPreLoadModel(path):
    resultData = {}
    resultData['isSuccess'] = 'true'
    resultData['result'] = {}
    resultData['ErrorMsg'] = ""
    modelDict = {}
    extension = '.hdf5'
    modelname = ''
    char_list = ''
    try:
        for filename in os.listdir(path):
            if filename.endswith('.hdf5'):
                modelName = os.path.splitext(filename)[0]
        fin = open(path + modelName + '.txt', 'rt')
        SetValue = fin.read().replace('\n','')
        fin.close()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        char_list = SetValue

        inputs = Input(shape=(32, 128, 1))

        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
        
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

        outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)
        fullfilepath = os.path.join(path, modelName + extension)
        act_model = Model(inputs, outputs)
        
        act_model.load_weights(fullfilepath)
        modelDict['act_model'] = act_model
        modelDict['char_list'] = char_list
        resultData['result'] = modelDict
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0]  # 取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        resultData['isSuccess'] = 'false'
        resultData['ErrorMsg'] = str(errorMsg)
    return resultData
