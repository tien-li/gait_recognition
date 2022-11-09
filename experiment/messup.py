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
train_data = pd.read_excel('/Users/ken/Desktop/專題/exp3/all_mess.xlsx', sheet_name = 'x')
label_data = pd.read_excel('/Users/ken/Desktop/專題/exp3/all_mess.xlsx', sheet_name = 'y')
train_data = np.array(train_data)
label_data = np.array(label_data)
x_train,x_vali,y_train,y_vali=train_test_split(train_data,label_data,test_size=0.01)

from pandas import DataFrame
F = DataFrame(x_train)
F.to_excel('/Users/ken/Desktop/專題/exp3/all_messupx.xlsx',sheet_name='x')

D = DataFrame(y_train)
D.to_excel('/Users/ken/Desktop/專題/exp3/all_messup.xlsx',sheet_name='y')

