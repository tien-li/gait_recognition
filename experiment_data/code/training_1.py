#!/usr/bin/env python
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
testing_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup.xlsx', sheet_name = 'x')
testing_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup.xlsx', sheet_name = 'y')
train_data = np.array(train_data)
label_data = np.array(label_data)
testing_x  = np.array(testing_x)
testing_y  = np.array(testing_y)
x_train,x_vali,y_train,y_vali=train_test_split(train_data,label_data,test_size=0.15)
#print(x_vali)

step = [280]
for j in step:
    model=Sequential()
    model.add(Dense(input_dim=1*100,units=j,activation='elu'))
    model.add(Dropout(0.15))
    for i in range(1,7):
        model.add(Dense(units=j,activation='elu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=j,activation='elu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=j,activation='elu'))
        model.add(Dropout(0.15))
    model.add(Dense(units=4,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',f1])
    model.summary()

    history=model.fit(x_train,y_train,batch_size = 128,epochs=100) 

    #plt.plot(history.history['acc'],label='acc')
    #plt.plot(history.history['loss'],label='loss')
    #plt.plot(history.history['f1'],label='f1')
    #plt.xlabel('Epochs')

    print('\n------------------Validation data------------------')
    result_vali=model.evaluate(x_vali,y_vali)
    #print(result_vali)
    F = DataFrame(result_vali)
    #F.to_excel('/Users/ken/Desktop/x_predict.xlsx')
    G = DataFrame(y_vali)
    #G.to_excel('/Users/ken/Desktop/y_vali.xlsx')
    #print('\nlayer:',j*3-1)
    print('\nTotal loss on Validation Set:',result_vali[0])
    print('\nAccuracy of Validation Set:',result_vali[1])
    print(result_vali[2])
    #print('\nF1 of Validation Set:',result_vali[2],'\n')
    print('\n')
    print('\n------------------testing data------------------')
    result_test=model.evaluate(testing_x,testing_y)
    print('\nTotal loss on Prediction Set:',result_test[0])
    print('\nAccuracy of Prediction Set:',result_test[1])
    print(result_test[2])
    #print('\nF1 of Prediction Set:',result_test[2],'\n')

#model.save('CNN_model.h5')