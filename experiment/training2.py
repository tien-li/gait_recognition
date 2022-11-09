import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam,Adagrad
from sklearn.model_selection import cross_val_score
from keras.callbacks import History
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import EarlyStopping,ModelCheckpoint
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

train_data = pd.read_excel('/Users/ken/Desktop/專題/trainingdata4.xlsx', sheet_name = 'train')
label_data = pd.read_excel('/Users/ken/Desktop/專題/trainingdata4.xlsx', sheet_name = 'label')
train_data = np.array(train_data)
label_data = np.array(label_data)
x_train,x_vali,y_train,y_vali = train_test_split(train_data,label_data,test_size=0.1)
model = Sequential()
#model_m.add(Reshape((x_train, y_train), input_shape=(x_train, y_train)))
x_train = x_train.reshape((2914, 100, 1))

model.add(Conv1D(25, 5, activation='elu', input_shape=(100, 1)))
model.add(Conv1D(25, 5, activation='elu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.1))
model.add(Conv1D(25, 5, activation='elu'))
model.add(Conv1D(25, 5, activation='elu'))
model.add(MaxPooling1D())
model.add(Dropout(0.1))
model.add(Conv1D(25, 5, activation='elu'))
model.add(Conv1D(25, 5, activation='elu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.1))
model.add(Dense(input_dim=1*20,units=120,activation='elu'))
for i in range(1,10):
    model.add(Dense(units=120,activation='elu'))
    model.add(Dropout(0.1))
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])

print(model.summary())

history=model.fit(x_train, y_train, batch_size=256, epochs=150) 

model.save('/Users/ken/Desktop/專題/CNN_model11.h5')