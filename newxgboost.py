import h5py
import numpy as np
import random
import gc
import xgboost as xgb
from xgboost import XGBClassifier
from keras.utils import to_categorical
import csv
import os
#from readData_1 import *
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, ShuffleSplit
import pickle
import matplotlib.pylab as plt
import pandas as pd
import time
import datetime

# train_data_file = "./val_data.npy"
# train_label_file = "./val_label.npy"
train_label_file = "./newdata/train_data_8000_label_sen2.npy"
train_data_file = "./newdata/train_data_8000_data_sen2.npy"
val_data_file = "./newdata/validation_data_sen2.npy"
val_label_file = "./newdata/validation_label_sen2.npy"
test_data_file = "./newdata/test_sen2.npy"

train_data = np.load(train_data_file)
train_label = np.load(train_label_file)
val_data = np.load(val_data_file)
val_label = np.load(val_label_file)
# test1_data = np.load('./test_data.npy')
test_data = np.load(test_data_file)


def save_result(save_name, label):

    with open(save_name, 'w') as f:
        for item in label:
            item = int(item)
            one_hot_label = np.zeros(17)
            one_hot_label[item] = 1
            str_label = str(one_hot_label).replace(
                "[", "").replace('.]', '').replace('. ', ",")+"\n"

            f.write(str_label)


def xgb_classify():
    global train_data, train_label, test_data, val_data, val_label, test_data
    xgb_model = None
    kf = KFold(n_splits=2, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bytree=0.8, gamma=0.01, gpu_id=0, eval_metric='merror',
                                  importance_type='weight', learning_rate=0.1, max_delta_step=0,
                                  max_depth=5, min_child_weight=1, missing=None, n_estimators=500,
                                  n_jobs=1, nthread=-1, num_class=17, objective='multi:softmax',
                                  random_state=0, reg_alpha=0.001, reg_lambda=5, scale_pos_weight=1,
                                  seed=0, silent=0, subsample=0.5, tree_method='gpu_hist')
        xgb_model.fit(x_train, y_train, eval_metric='merror', verbose=True,
                      eval_set=[(x_test, y_test)], early_stopping_rounds=100)
        predictions = xgb_model.predict(x_test)
        print(confusion_matrix(y_test, predictions))

    xgb_model.fit(data, label, eval_metric='merror', verbose=True)

    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    save_pkl_name = "./pkl/best_clf_"+ time_str+".pkl"
    pickle.dump(xgb_model, open(save_pkl_name))

    prediction = xgb_model.predict(data)
    val_prediction = xgb_model.predict(val_data)
    test_prediction = xgb_model.predict(test_data)
    print("prediction")
    print(prediction)
    print("val_prediction")
    print(val_prediciton)
    print("test_prediction")
    print(test_prediction)
    train_acc = metrics.accuracy_score(label, train_prediction)
    val_acc = metrics.accuracy_score(val_label, val_prediction)
    print("train accuracy: ",train_acc)
    print("val_accuracy: ", val_acc)
    train_confusion = confusion_matrix(label, train_prediction)
    val_confusion = confusion_matrix(val_label, val_prediction)
    print("saving confusion...")
    train_confusion_file_name = "./confusion/train_conv_"+time_str+".npy"
    val_confusion_file_name = "./confusion/val_conv_"+time_str+".npy"
    np.save(train_confusion_file_name, train_confusion)
    np.save(val_confusion_file_name, val_confusion)
    print("train confusion:")
    print(train_confusion)
    print("val confusion:")
    print(val_confusion)   
    print("saving result") 
    result_save_path = "./result/result_"+time_str+".csv"
    save_result(result_save_path, test_prediction)
    return train_acc, val_acc, test_prediction


def gridsearch(train_data, train_label, val_data, val_label, test_data):
    print("grid search")
    xgb_model = None
    xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bytree=0.8, gamma=0.01, gpu_id=0, eval_metric='merror',
                              importance_type='weight', learning_rate=0.3, max_delta_step=0,
                              min_child_weight=1, missing=None, n_estimators=100,
                              n_jobs=1, nthread=-1, num_class=17, objective='multi:softmax',
                              random_state=0, reg_alpha=0.001, reg_lambda=10, scale_pos_weight=1,
                              seed=0, silent=1, subsample=0.7, tree_method='gpu_hist')
    params_grid = {
        "max_depth":[5,7],
    }
    gridclfer = GridSearchCV(xgb_model, param_grid=params_grid,verbose=10,cv=5,n_jobs=1,iid=False,scoring='accuracy')
    # gridclfer.fit(train_data, train_label, eval_metric='merror', verbose=True,
    #                   eval_set=[(train_data, train_label)], early_stopping_rounds=20)
    gridclfer.fit(train_data, train_label, eval_metric='merror', verbose=10,eval_set=[(train_data,train_label)],)
    best_est = gridclfer.best_estimator_
    best_params = gridclfer.best_params_

    print("best_est:")
    print(best_est)
    print("best_params:")
    print(best_params)
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    save_pkl_name = "./pkl/best_clf_"+ time_str+".pkl"
    pickle.dump(gridclfer, open(save_pkl_name))

    train_prediction = best_est.predict(train_label)
    val_prediction = best_est.predict(val_data)
    test_prediction = best_est.predict(test_data)
    train_acc = metrics.accuracy_score(train_label, train_prediction)
    val_acc = metrics.accuracy_score(val_label, val_prediction)
    print("train accuracy: ",train_acc)
    print("val_accuracy: ", val_acc)
    train_confusion = confusion_matrix(train_label, train_prediction)
    val_confusion = confusion_matrix(val_label, val_prediction)
    print("saving confusion...")
    train_confusion_file_name = "./confusion/train_conv_"+time_str+".npy"
    val_confusion_file_name = "./confusion/val_conv_"+time_str+".npy"
    np.save(train_confusion_file_name, train_confusion)
    np.save(val_confusion_file_name, val_confusion)
    print("train confusion:")
    print(train_confusion)
    print("val confusion:")

    print(val_confusion)   
    print("saving result")  
    result_save_path = "./result/result_"+time_str+".csv"
    save_result(result_save_path, test_prediction)
    
def xgbmodelc(train_data, train_label, val_data, val_label, test_data):
    # global train_data, trainlabel, val_data, val_label, test_data  
    xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bytree=0.8, gamma=0, eval_metric='merror',
                              importance_type='weight', learning_rate=0.3, max_delta_step=0,
                              max_depth=7, min_child_weight=1, missing=None, n_estimators=2000,
                              n_jobs=-1, nthread=-1, num_class=17, objective='multi:softmax',
                              random_state=0, reg_alpha=0.001, reg_lambda=1, scale_pos_weight=1,
                              seed=0, silent=True, subsample=0.8)
    # xgb_model.fit(data, label,eval_set=())
    xgb_model.fit(train_data, train_label, eval_metric='merror', verbose=True,
                      eval_set=[(train_data, train_label),(val_data, val_label)], early_stopping_rounds=100)
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    # save_pkl_name = "./pkl/best_clf_"+ time_str+".pkl"
    # pickle.dump(xgb_model, open(save_pkl_name))

    train_prediction = xgb_model.predict(train_data)
    val_prediction = xgb_model.predict(val_data)
    test_prediction = xgb_model.predict(val_data)
    print("train_prediction")
    print(train_prediction)
    print("val_prediction")
    print(val_prediction)
    print("test_prediction")
    print(test_prediction)
    train_acc = metrics.accuracy_score(train_label, train_prediction)
    val_acc = metrics.accuracy_score(val_label, val_prediction)
    print("train accuracy: ",train_acc)
    print("val_accuracy: ", val_acc)
    train_confusion = confusion_matrix(train_label, train_prediction)
    val_confusion = confusion_matrix(val_label, val_prediction)
    print("saving confusion...")
    train_confusion_file_name = "./confusion/train_conv_"+time_str+".npy"
    val_confusion_file_name = "./confusion/val_conv_"+time_str+".npy"
    np.save(train_confusion_file_name, train_confusion)
    np.save(val_confusion_file_name, val_confusion)
    print("train confusion:")
    print(train_confusion)
    print("val confusion:")
    print(val_confusion)   
    print("saving result") 
    result_save_path = "./result/result_"+time_str+".csv"
    save_result(result_save_path, test_prediction)
    return train_acc, val_acc, test_prediction

def multiTrain(data, label, test_data):

    best_train_acc = 0
    best_val_acc = 0
    result_list = []
    train_acc_list = []
    val_acc_list = []
    best_epoch = None
    final_result = []
    splitstate = ShuffleSplit(n_splits=18, test_size=.20)
    splitstate.get_n_splits(data, label)
    epoch = 0
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    for train_index, val_index in splitstate.split(data, label):
        print("epoch: ", epoch+1)
        epoch+=1
        print("TRAIN:", train_index, "TEST: ", val_index)
        sub_train_data = np.array([train_data[i] for i in train_index])
        sub_train_label = np.array([train_label[i] for i in train_index])
        sub_val_data = np.array([train_data[i] for i in val_index])
        sub_val_label = np.array([train_label[i] for i in val_index])
        train_acc, val_acc, test_prediction = xgbmodelc(sub_train_data, sub_train_label, sub_val_data, sub_val_label, test_data)
        if val_acc > best_val_acc:
            print("find a better val_acc: "+str(best_val_acc)+" -> "+str(val_acc))
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch
        result_list.append(test_prediction)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    print("best_epoch: {}".format(epoch))
    best_result = result_list[epoch-1]
    best_result_save_name = "./best/best_result_"+time_str+".csv"
    save_result(best_result_save_name, best_result)
    print(best_val_acc)
    print(train_acc_list)
    result_list = np.array(result_list)
    for i in range(17):        
        counts = np.bincount(result_list[:, i])
        index = np.argmax(counts)
        final_result.append(index)
    final_result = np.array(final_result)
    one_hots = to_categorical(final_result).astype(np.int32)
    csvfile = open('rank_result_1.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    for i in one_hots:
        writer.writerow(i)
    csvfile.close()
    print("预测完毕!")


        
        
def main():
    train_label_file = "./train_data_8000_label.npy"
    train_data_file = "./train_data_8000_data.npy"
    val_data_file = "./val_data.npy"
    val_label_file = "./val_label.npy"
    test_data_file = "./testb_data.npy"

    train_data = np.load(train_data_file)
    train_label = np.load(train_label_file)
    val_data = np.load(val_data_file)
    val_label = np.load(val_label_file)
    test_data = np.load('./test_data.npy')
    testb_data = np.load(test_data_file)

     
   # print(label)
    epoch = 1
    best_train_acc = 0
    best_val_acc = 0
    result_list = []
    train_acc_list = []
    val_acc_list = []
    best_epoch = None
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))

    for i in range(epoch):

        print("epoch:", str(i+1))
        train_acc, val_acc, result = xgbmodelc()
        if val_acc > best_val_acc:
            print("find a better val_acc: "+str(best_val_acc)+" -> "+str(val_acc))
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = i+1
        result_list.append(result)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    print("best_epoch: {}".format(i))
    best_result = result_list[i-1]
    best_result_save_name = "./best/best_result_"+time_str+".csv"
    save_result(best_result_save_name, best_result)
    print(best_val_acc)
    print(train_acc_list)
train_label = np.array([[i] for i in train_label])

train_data = np.concatenate((train_data, train_label), axis=1)
np.random.shuffle(train_data)
train_label = train_data[:,-1]
train_data = train_data[:,:-1]
# gridsearch(train_data, train_label, val_data, val_label, test_data)
# xgbmodelc(train_data, train_label, val_data, val_label, test_data)
multiTrain(train_data, train_label, test_data)