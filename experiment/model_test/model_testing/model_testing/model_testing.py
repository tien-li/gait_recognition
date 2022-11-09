print("step0")
#設定所需套件
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Activation,Dense,Dropout
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam,Adagrad
from sklearn.model_selection import cross_val_score
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import EarlyStopping,ModelCheckpoint
print("step1")

#Load Data
test1_data = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test1_train.xlsx',sheet_name = 'test1')
test1_label = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test1_train.xlsx',sheet_name = 'label')
test2_data = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test2_train.xlsx',sheet_name = 'test2')
test2_label = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test2_train.xlsx',sheet_name = 'label')
test3_data = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test3_train.xlsx',sheet_name = 'test3')
test3_label = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test3_train.xlsx',sheet_name = 'label')
test4_data = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test4_train.xlsx',sheet_name = 'test4')
test4_label = pd.read_excel(r'C:\Users\user\Downloads\台大\108學年-下\專題研究\ML資料\test_data\test4_train.xlsx',sheet_name = 'label')
print("step2")
test1_data = np.array(test1_data)
test1_label = np.array(test1_label)
test2_data = np.array(test2_data)
test2_label = np.array(test2_label)
test3_data = np.array(test3_data)
test3_label = np.array(test3_label)
test4_data = np.array(test4_data)
test4_label = np.array(test4_label)

print("step3")
#Define F1 score function
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

print("step4")
#Load model
model = load_model('model_lin_huang_wu3.h5',custom_objects={'f1': f1})

print("step5")
#model testing
print('\n------------------testing data------------------')

#print('\n------------------test1------------------')
#result_test1 = model.evaluate(test1_data,test1_label)
#print('\n  Total loss on Prediction Set:',result_test1[0])
#print('\n  Accuracy of Prediction Set:',result_test1[1])
#print('\n  F1 of Prediction Set:',result_test1[2],'\n')

print('\n------------------test2------------------')
result_test2 = model.evaluate(test2_data,test2_label)
print('\n  Total loss on Prediction Set:',result_test2[0])
print('\n  Accuracy of Prediction Set:',result_test2[1])
print('\n  F1 of Prediction Set:',result_test2[2],'\n')

print('\n------------------test3------------------')
result_test3 = model.evaluate(test3_data,test3_label)
print('\n  Total loss on Prediction Set:',result_test3[0])
print('\n  Accuracy of Prediction Set:',result_test3[1])
print('\n  F1 of Prediction Set:',result_test3[2],'\n')

print('\n------------------test4------------------')
result_test4 = model.evaluate(test4_data,test4_label)
print('\n  Total loss on Prediction Set:',result_test4[0])
print('\n  Accuracy of Prediction Set:',result_test4[1])
print('\n  F1 of Prediction Set:',result_test4[2],'\n')

print("done")