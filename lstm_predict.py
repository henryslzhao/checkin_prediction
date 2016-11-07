
from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import re
#import operator
#import helper
from collections import Counter
import sys
from keras.models import model_from_json
debug = True
pretrain = False
num_debug_user = 1000

def load_model(folder):
    with open(folder+".json",'rb') as f:
        json_string   = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model

def load_data(fname,):
    checkin = open(fname, 'r')
    uid_seq = {}
    pois_index = {}
    index_poi = {}
    while True:
        line = checkin.readline().strip()
        if not line:
            break
        uid = int(line)
        num = int(checkin.readline().strip())
        count = 0
        seq = [] 
        if debug and uid==num_debug_user:
            break
        while count < num:
            line = checkin.readline().strip()
            line = line.split(',')
            poi = int(line[-1])
            if poi not in pois_index:
                index = len(pois_index)
                pois_index[poi] = index
                index_poi[index] = poi
            seq.append(pois_index[poi])
            count += 1
        uid_seq[uid] = seq
    return uid_seq, pois_index, index_poi

def get_train_test(uid_seq, vocSize, maxlen=5, rate=0.2):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for uid in uid_seq:
        seq = uid_seq[uid]
        for i in xrange(len(seq)):
            if i+maxlen < len(seq):
                x = seq[i:i+maxlen]
                y = seq[i+maxlen]
                if np.random.rand() < rate:
                    x_test.append(x)
                    y_test.append(y)
                else:
                    x_train.append(x)
                    y_train.append(y)
            else:
                break
                    
    tmp_y_train = np.zeros((len(x_train), vocSize))
    for i in xrange(len(y_train)):
        tmp_y_train[i][y_train[i]] = 1
    tmp_y_test = np.zeros((len(x_test), vocSize))
    for i in xrange(len(y_test)):
        tmp_y_test[i][y_test[i]] = 1
    x_train = np.array(x_train)
    y_train = tmp_y_train
    x_test = np.array(x_test)
    y_test = tmp_y_test
    
    return x_train, y_train, x_test, y_test
    
print("loading data")
maxlen = 5
batch_size = 10

data, pois_index, index_poi = load_data('fsCheckin.csv')
vocSize = len(pois_index)
x_train, y_train, x_test, y_test = get_train_test(data, vocSize, maxlen)


print('xtrain.shape', (x_train.shape))
print('ytrain.shape', (y_train.shape))
print('xtest.shape', x_test.shape)
print('ytest.shape', y_test.shape)
print('num of pois', vocSize)

if pretrain:
    model = load_model('lstm_predict')
else:
    model = Sequential()
    model.add(Embedding(vocSize, 128, input_length=maxlen, dropout=0.5))
    model.add(LSTM(128, dropout_W=0.5, dropout_U=0.5))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
#model.compile(loss='binary_crossentropy',optimizer='adam',class_mode="binary")

    model.add(Dense(vocSize))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


wordModel = Sequential()
wordModel.add(Embedding(vocSize, 256, input_length=maxlen))
wordModel.add(Dropout(0.2))
wordModel.add(LSTM(512, return_sequences=True))    
wordModel.add(Dropout(.2))
wordModel.add(LSTM(512, return_sequences=False))
    #wordModel.add(RepeatVector(maxlen))
    
wordModel.add(Dense(vocSize))
wordModel.add(Activation('softmax'))
wordModel.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Train...")
num_epoch = 15
epoch = 0
while epoch < num_epoch:
    epoch += 1
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,validation_data=(x_test, y_test))

#sava model
    if not pretrain:
        jsonstring = model.to_json()
        with open("lstm_predict.json",'wb') as f:
            f.write(jsonstring)
            model.save_weights("lstm_predict.h5",overwrite=True)
    y_predict = model.predict(x_test)
    num_correct = 0
    temp_y_predict = np.zeros_like(y_predict)
    for i in xrange(len(y_predict)):
        y_poi = np.random.choice(range(vocSize), p=y_predict[i])
        if y_test[i][y_poi]:
            num_correct += 1
#        temp_y_predict[i][y_poi] = 1
#correct = np.sum(np.dot(y_test, np.transpose(temp_y_predict)))
#    print(correct)
    print('num_corect: ', num_correct)
    print(num_correct*1.0/y_predict.shape[0])
