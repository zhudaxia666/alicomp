import h5py
import numpy as np
import random
import gc
import xgboost as xgb
from xgboost import XGBClassifier
# from xgboost.sklearn import XGBClassifier
# from keras.utils import to_categorical
import csv
import os
from readData_1 import *
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import pickle
import matplotlib.pylab as plt
import pandas as pd
import time
DataSet = readData()
all_train_length = DataSet.get_length()
label = []
for index in range(all_train_length):
    # data.append(DataSet.get_ND_mat(index).flatten())
    # print(DataSet.get_op_label(index)[0])
    label.append(DataSet.get_label(index))
# data = np.array(data)
label = np.array(label)
print(label.shape)
print(np.sum(label, axis=0))