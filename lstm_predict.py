
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

def load_data(fname):
    checkin = open(fname, 'r')
    
    
print("loading data")



model = Sequential()
model.add(Embedding(vocSize, 128, input_length=maxlen, dropout=0.5))
model.add(LSTM(128, dropout_W=0.5, dropout_U=0.5))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")

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
