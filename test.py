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
# import matplotlib.pylab as plt
import pandas as pd
import time


DataSet = readData()
all_train_length = DataSet.get_length()
all_val_length = DataSet.get_length(train=False)
all_testb_length = DataSet.get_length(test=True)
all_test_length = DataSet.get_length(test=True, label=1)
print("train data: "+str(all_train_length))
print("test data : "+str(all_test_length))
print("val data: " + str(all_val_length))
print("testb data: "+str(all_testb_length))

data = []
label = []
test_data = []
val_data = []
val_label = []
testb_data = []


#for index in range(all_val_length):
#      data.append(DataSet.get_ND_mat(index).flatten())
#      # print(DataSet.get_op_label(index)[0])
     #label.append(DataSet.get_op_label(index,train=False)[0])
#  data = np.array(data)
#label = np.array(label)

# data = np.concatenate((data, label), axis=1)

# np.random.shuffle(data)
#
# label = data[:,-1]
# data = data[:,:-1]

for index in range(all_test_length):
    test_data.append(DataSet.get_ND_mat(index, test=True, label=1).flatten())

test_data = np.array(test_data)

# np.save('./train_data_8000_label_sen2.npy', label)
# np.save('./validation_data_sen2.npy', data)
np.save('./validation_label_sen2.npy', label)
np.save('./test_sen2.npy', test_data)

# for index in range(all_val_length):
#     val_data.append(DataSet.get_ND_mat(index, train=False).flatten())
#     val_label.append(DataSet.get_op_label(index, train=False)[0])

# val_data = np.array(val_data)
# val_label = np.array(val_label)

# for index in range(all_testb_length):
#     testb_data.append(DataSet.get_ND_mat(index, test=True).flatten())
# testb_data = np.array(testb_data)
# print(label.shape)
# print(data.shape)
# print(val_data.shape)
# print(val_label.shape)
# print(test_data.shape)
# print(testb_data.shape)

# np.save('./train_data_8000_data.npy', data)
# np.save('./train_data_8000_label.npy', label)
# np.save('./val_data.npy', val_data)
# np.save('./val_label.npy', val_label)
# np.save('./test_data.npy', test_data)
# np.save('./testb_data.npy', testb_data)

# data = np.load('./train_data_8000_data.npy')
# label = np.load('./train_data_8000_label.npy')
# val_data = np.load('./val_data.npy')
# val_label = np.load('./val_label.npy')
# test_data = np.load('./test_data.npy')
# testb_data = np.load('./testb_data.npy')

data = np.load('./validation_data_sen2.npy')
label = np.load('./validation_label_sen2.npy')
test = np.load('./test_sen2.npy')
print(data.shape)
print(label.shape)
print(label)
print(test.shape)
print("============数据加载完毕===========")
'''

def xgb_classify():
    global data, label, test_data, val_data, val_label, testb_data
    xgb_model = None
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bytree=0.8, gamma=0.01, gpu_id=0,eval_metric='merror',
                                  importance_type='weight', learning_rate=0.1, max_delta_step=0,
                                  max_depth=5, min_child_weight=1, missing=None, n_estimators=500,
                                  n_jobs=1, nthread=-1, num_class=17, objective='multi:softmax',
                                  random_state=0, reg_alpha=0.001, reg_lambda=5, scale_pos_weight=1,
                                  seed=0, silent=0, subsample=0.7, tree_method='gpu_hist')
        xgb_model.fit(x_train, y_train, eval_metric='merror', verbose=True,
                  eval_set=[(x_test, y_test)], early_stopping_rounds=100)
        predictions = xgb_model.predict(x_test)
        print(confusion_matrix(y_test, predictions))

    # xgb_model.fit(data, label, eval_metric='merror', verbose=True)

    # pkl_name = "best_boston"+"_"
    pickle.dump(xgb_model, open("best_boston_8.pkl", "wb"))

    # Roc AUC with all train data
    prediction_prob = xgb_model.predict_proba(data)
    prediction = xgb_model.predict(data)
    val_prediction_prob = xgb_model.predict_proba(val_data)
    val_prediction = xgb_model.predict(val_data)
    print('prediction_prob:')
    print(prediction_prob)
    print('val_prediction_prob:')
    print(val_prediction_prob)
    # print("Roc AUC : ", metrics.roc_auc_score(label, prediction_prob[:,1], average='macro'))
    print("Accuracy: ", metrics.accuracy_score(label, prediction))
    print(confusion_matrix(label, prediction))

    print("val Accuracy: ", metrics.accuracy_score(val_label, val_prediction))
    print(confusion_matrix(val_label, val_prediction))

    print("test")
    probs_test = xgb_model.predict_proba(test_data)
    label_test = np.argmax(probs_test, axis=1)
    # print("probs_test:")
    # print(probs_test)
    print("label_test:")
    print(label_test)
    print(label_test.shape)

    with open('./result_all_old_9.csv', 'w') as f:

        for item in label_test:
            one_hot_label = np.zeros(17)
            one_hot_label[item] = 1
            str_label = str(one_hot_label).replace(
                "[", "").replace('.]', '').replace('. ', ",")+"\n"

            f.write(str_label)

    print("test1")
    probs1 = xgb_model.predict_proba(testb_data)
    label1 = np.argmax(probs1, axis=1)
    print(label1.shape)
    print(label1)
    with open('./result_all_new_9.csv', 'w') as f1:

        for item in label1:
            one_hot_label = np.zeros(17)
            one_hot_label[item] = 1
            str_label = str(one_hot_label).replace(
                "[", "").replace('.]', '').replace('. ', ",")+"\n"

            f1.write(str_label)

xgb_classify()

'''