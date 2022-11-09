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

testing1_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test1/test1_train.xlsx', sheet_name = 'test1')
testing1_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test1/test1_train.xlsx', sheet_name = 'label')
testing1_x  = np.array(testing1_x)
testing1_y  = np.array(testing1_y)
testing1_x = testing1_x.reshape((655, 100, 1))

testing2_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test2/test2_train.xlsx', sheet_name = 'test2')
testing2_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test2/test2_train.xlsx', sheet_name = 'label')
testing2_x  = np.array(testing2_x)
testing2_y  = np.array(testing2_y)
testing2_x = testing2_x.reshape((598, 100, 1))

testing3_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test3/test3_train.xlsx', sheet_name = 'test3')
testing3_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test3/test3_train.xlsx', sheet_name = 'label')
testing3_x  = np.array(testing3_x)
testing3_y  = np.array(testing3_y)
testing3_x = testing3_x.reshape((547, 100, 1))

testing4_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test4/test4_train.xlsx', sheet_name = 'test4')
testing4_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/test4/test4_train.xlsx', sheet_name = 'label')
testing4_x  = np.array(testing4_x)
testing4_y  = np.array(testing4_y)
testing4_x = testing4_x.reshape((542, 100, 1))

messup_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup.xlsx', sheet_name = 'x')
messup_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup.xlsx', sheet_name = 'y')
messup_x  = np.array(messup_x)
messup_y  = np.array(messup_y)
messup_x = messup_x.reshape((648, 100, 1))

messup2_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup2.xlsx', sheet_name = 'x')
messup2_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup2.xlsx', sheet_name = 'y')
messup2_x  = np.array(messup2_x)
messup2_y  = np.array(messup2_y)
messup2_x = messup2_x.reshape((597, 100, 1))

messup3_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup3.xlsx', sheet_name = 'x')
messup3_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup3.xlsx', sheet_name = 'y')
messup3_x  = np.array(messup3_x)
messup3_y  = np.array(messup3_y)
messup3_x = messup3_x.reshape((546, 100, 1))

messup4_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup4.xlsx', sheet_name = 'x')
messup4_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/messup4.xlsx', sheet_name = 'y')
messup4_x  = np.array(messup4_x)
messup4_y  = np.array(messup4_y)
messup4_x = messup4_x.reshape((541, 100, 1))

all_messup_x  = pd.read_excel('/Users/ken/Desktop/專題/exp3/all_messup.xlsx', sheet_name = 'x')
all_messup_y  = pd.read_excel('/Users/ken/Desktop/專題/exp3/all_messup.xlsx', sheet_name = 'y')
all_messup_x  = np.array(all_messup_x)
all_messup_y  = np.array(all_messup_y)
all_messup_x = all_messup_x.reshape((2306, 100, 1))

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

model = load_model('/Users/ken/Desktop/專題/CNN_model3.h5', custom_objects = {'f1':f1})

print('\n------------------testing1 data------------------')
result_test=model.evaluate(testing1_x,testing1_y)
predict_test=model.predict(testing1_x)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
from pandas import DataFrame
F = DataFrame(predict_test)
F.to_excel('/Users/ken/Desktop/專題/exp3/predict_test1.xlsx')
print('\n')
print('\n------------------testing2 data------------------')
result_test=model.evaluate(testing2_x,testing2_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------testing3 data------------------')
result_test=model.evaluate(testing3_x,testing3_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------testing4 data------------------')
result_test=model.evaluate(testing4_x,testing4_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------messup data------------------')
result_test=model.evaluate(messup_x,messup_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------messup2 data------------------')
result_test=model.evaluate(messup2_x,messup2_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------messup3 data------------------')
result_test=model.evaluate(messup3_x,messup3_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------messup4 data------------------')
result_test=model.evaluate(messup4_x,messup4_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])
print('\n')
print('\n------------------all_messup data------------------')
result_test=model.evaluate(all_messup_x,all_messup_y)
print('\nTotal loss on Prediction Set:',result_test[0])
print('\nAccuracy of Prediction Set:',result_test[1])
print('\nAccuracy of f1:',result_test[2])