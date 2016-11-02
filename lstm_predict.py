
from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import re
import operator#
#import helper
from collections import Counter
import sys
from keras.models import model_from_json
