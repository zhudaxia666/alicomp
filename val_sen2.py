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
from readData import *
from sklearn.model_selection import GridSearchCV,cross_val_score  
from sklearn import  metrics
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import pickle
import matplotlib.pylab as plt  
import pandas as pd
import time
DataSet = readData()
# all_train_length = DataSet.get_length()
# all_val_length = DataSet.get_length(train=False)
# all_test_length = DataSet.get_length(test=True)
# all_testb_length = DataSet.get_length(test=True,label=1)
# print("train data: "+str(all_train_length))
# print("test data : "+str(all_test_length))
# print("val data: "+ str(all_val_length))
# print("testb data: "+str(all_testb_length))

# data = []
# label = []
# test_data = []
# val_data = []
# val_label = []
# testb_data = []

# for index in range(all_train_length):
#     data.append(DataSet.get_ND_mat(index).flatten())
#     # print(DataSet.get_op_label(index)[0])
#     label.append(DataSet.get_op_label(index)[0])
# data = np.array(data)
# label = np.array(label)

# for index in range(all_test_length):
#     test_data.append(DataSet.get_ND_mat(index,test=True).flatten())

# test_data = np.array(test_data)

# for index in range(all_val_length):
#     val_data.append(DataSet.get_ND_mat(index,train=False).flatten())
#     val_label.append(DataSet.get_op_label(index,train=False)[0])

# val_data = np.array(val_data)
# val_label = np.array(val_label)

# for index in range(all_testb_length):
#     testb_data.append(DataSet.get_ND_mat(index, test=True,label=1).flatten())
# testb_data = np.array(testb_data)
# print(label.shape)
# print(data.shape)
# print(val_data.shape)
# print(val_label.shape)
# print(test_data.shape)
# print(testb_data.shape)
data = np.load('./train_data_8000_data_1.npy')
label = np.load('./train_data_8000_label_1.npy')
val_data = np.load('./val_data_1.npy')
val_label = np.load('./val_label_1.npy')
test_data = np.load('./test_data_1.npy')
testb_data = np.load('./testb_data_1.npy')


# base_dir = "data"
# path_training = os.path.join(base_dir, "training.h5")
# path_test = os.path.join(base_dir, "round1_test_a_20181109.h5")
# path_train = os.path.join(base_dir, "new_train8000.h5")
# path_validation = os.path.join(base_dir,"validation/validation.h5")
# fid_train = h5py.File(path_train,'r')
# fid_test = h5py.File(path_test, 'r')
# fid_training = h5py.File(path_test, 'r')


# fid_val = h5py.File(path_validation,'r')
# test_s1=fid_training['sen1']
# test_s2=fid_training['sen2']
# train_label=fid_training['label']
# def Test_data():
# #将sen2的数据按照2轴
#     s1 = fid_training['sen1']
#     s2 = fid_training['sen2']
#     data = []
#     for i in range(len(s2)):
#         part = []
#         for chanel1 in range(s1.shape[3]):
#             for j1 in range(s1.shape[1]):
#                 part.extend(s1[i, j1, :, chanel1])
#         for chanel2 in range(s2.shape[3]):
#             for j2 in range(s2.shape[1]):
#                 part.extend(s2[i, j2, :, chanel2])
#         data.append(part)
#     print("数据转换成功")
#     return np.array(data)

def modelMetrics(clf,train_x,train_y,isCv=True,cv_folds=5,early_stopping_rounds=50):
    if isCv:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x,label=train_y)
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=clf.get_params()['n_estimators'],nfold=cv_folds,
                          metrics='auc',early_stopping_rounds=early_stopping_rounds)#是否显示目前几颗树额
        clf.set_params(n_estimators=cvresult.shape[0])
 
    clf.fit(train_x,train_y,eval_metric='auc')
 
    #预测
    train_predictions = clf.predict(train_x)
    train_predprob = clf.predict_proba(train_x)[:,1]#1的概率
 
    #打印
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))
 
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar',title='Feature importance')
    plt.ylabel('Feature Importance Score')

def get_all_data(counts,tr_s1,tr_s2):
    s1_data = tr_s1
    s2_data = tr_s2

    res_list = []
    for item in range(counts):
        per_item = []
        this_item_1 = s1_data[int(item)]
        this_item_2 = s2_data[int(item)]
        for num in range(0, 8):
            per_item.append(this_item_1[:, :, num])
        for num in range(0, 10):
            per_item.append(this_item_2[:, :, num])
        res_list.append(np.array(per_item))
    res_mat = np.array(res_list)
    return res_mat


def get_label(counts, te_label):
    label = te_label
    res_list = []
    for item in range(counts):
        # per_item = []
        this_item = label[int(item)]
        res_list.append(this_item)
    res_mat = np.array(res_list)
    return res_mat


def write_h5(sen1,sen2, label, filename):
    # sen1 = np.zeros((shape,32,32,8))
    # sen2 = np.zeros((shape,32,32,10))
    # label= np.zeros((shape,17))
    f = h5py.File(filename, 'w')  # 创建一个h5文件，文件指针是f
    f['sen1'] = sen1
    f['sen2'] = sen2  # 将数据写入文件的主键data下面
    f['label'] = label
    print("写入完毕")  # 将数据写入文件的主键labels下面
    f.close()  # 关闭文件


def onehot_tranIndex(label):
    index_set = []
    # label=list(label)
    for i in range(len(label)):
        item = list(label[i])
        index_set.append(item.index(1))
    # write_csvfile(index_set)
    # print(index_set)
    return np.array(index_set)

xgb.XGBClassifier

def xgboost_class(train_data, train_label, valid_data, valid_label, test_data):
    params = {
        'booster': 'gbtree',
        # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
        'objective': 'multi:softmax',
        'num_class': 17,  # 类数，与 multisoftmax 并用
        'gamma': 0.01,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
        'max_depth': 8,  # 构建树的深度 [1:],典型值：3-10
        # 'lambda':450,  # L2 正则项权重
        'subsample': 0.5,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1],典型值：0.5-1
        'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1] 典型值：0.5-1
        # 'min_child_weight':12, # 节点的最少特征数
        'silent': 1,
        'eta': 0.32,  # 如同学习率0.3准确率0.63多
        'seed': 710,
        'nthread': 9,  # cpu 线程数,根据自己U的个数适当调整
        'tree_method': 'gpu_exact',

    }
    plst = list(params.items())
    num_rounds = 20  # 迭代你次数
    # xgtest = xgb.DMatrix(test)
    xgtrain = xgb.DMatrix(train_data, label=train_label)
    xgval = xgb.DMatrix(valid_data, label=valid_label)
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    # training model
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    # model.save_model('./model/xgb.model') # 用于存储训练出的模型
    preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    preds = [int(i) for i in preds]
    return preds
    # print(preds)
    # one_hots = to_categorical(preds).astype(np.int32)
    # print(one_hots)
    # csvfile=open('accuracy_22.csv','w',newline='')
    # write1=csv.writer(csvfile)
    # for i in one_hots:
    #     write1.writerow(i)
    # csvfile.close()
    # print("预测完毕!")


def result(train_sen1,train_label):
    tr_s1 = []
    # tr_s2 = []
    tr_label = []

    te_s1 = []
    # te_s2 = []
    te_label = []

    train = random.sample(range(len(train_label)), 20000)
    test = list(set(range(len(train_label))).difference(set(train)))
    # test=list(np.delete(range(125623),train))
    test = random.sample(test, 4000)

    for i in train:
        tr_s1.append(train_sen1[i])
        # tr_s2.append(train_sen2[i])
        tr_label.append(train_label[i])

    for k in test:
        te_s1.append(train_sen1[k])
        # te_s2.append(train_sen2[k])
        te_label.append(train_label[k])

    # train_data = get_all_data(10000,tr_s1,tr_s2).reshape(10000, -1)
    # # print(train_data.shape) #(8000, 18, 32, 32)
    # val_data = get_all_data(5000, te_s1,te_s2).reshape(5000, -1)

    train_label = get_label(20000, tr_label)
    train_label = onehot_tranIndex(train_label)
    val_label = get_label(4000, te_label)
    val_label = onehot_tranIndex(val_label)

    # test_data = Test_data()
    test_label = fid_test["sen1"]
    a = readData()
    data1 = []
    for i in range(len(test_label)):
        tem = a.get_ND_mat(i, None,1,8,True,False,True)
        data1.append(tem)
    test_data = np.array(data1).reshape(len(test_label), -1)

    item = xgboost_class(tr_s1, train_label, te_s1, val_label, test_data)
    print("预测完毕!")
    return item

def main():    
    file = h5py.File(path_train, 'r')
    train_label = file["label"]
    a=readData()
    data=[]
    for i in range(len(train_label)):
        tem=a.get_ND_mat(i,None)
        data.append(tem)
    data=np.array(data).reshape(len(train_label),-1)
    # print(data.shape)

    # file = h5py.File(path_train, 'r')
    # train_sen1 = file["sen1"]
    # train_sen2 = file["sen2"]
    # train_label = file["label"]

    part_data = []
    indexs = []

    for i in range(20):
        item = result(data, train_label)
        part_data.append(item)
        print("第%d训练完" % (i + 1))

    data = np.array(part_data)
    for i in range(len(part_data[0])):
        counts = np.bincount(data[:, i])
        index = np.argmax(counts)
        indexs.append(index)
    indexs = np.array(indexs)
    one_hots = to_categorical(indexs).astype(np.int32)
    csvfile = open('new_accuracy_val_1_4.csv', 'w', newline='')
    write1 = csv.writer(csvfile)
    for i in one_hots:
        write1.writerow(i)
    csvfile.close()
    print("预测完毕!")

def tun_parameters(train_x,train_y, params, params_grid):
    xgb1 = XGBClassifier(params)
    modelMetrics(xgb1,train_x,train_y)

'''
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

'''
def xgboost_classifier():
    global data, label, test_data, val_data, val_label, testb_data
    params = {
        
        'learning_rate':0.08,
        'n_estimators':1000,
        'nthread':10,
        'silent':0,
        'objective':'multi:softmax',
        'booster':'gbtree',
        'gamma':0.01,
        'colsample_bytree':0.8,
        # 'reg_alpha':0.005,
        'reg_lambda':1,
        'scale_pos_weight':1,
        'base_score':0.5,
        'seed':0,
        'random_state':0,
        'missing':None,
        'importance_type':'weight',
        'tree_method':'gpu_hist',
        'num_class':17,
        'gpu_id':0,
        'eval_metric':'merror',
        'subsample': 0.7,
        'max_delta_step':0,
        'reg_alpha':0.001
        
    }
    '''
        'max_depth': [3,4,5,6,7,8,9],
        'subsample': [0.6,0.7,0.8],
        'min_child_weight':[1,6,2],
        'max_delta_step': [0,1]
    '''
    '''
     'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
      'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
              'reg_alpha':[0.001,0.002,0.003,0.004,0.005]
    '''
    params_grid = {
        'max_depth': range(7,10,2),
        # 'subsample': [0.5,0.6,0.7,0.8,0.9,1.0],
        'min_child_weight':range(3,6,2),
        # 'max_delta_step': [0,1],
    }


    print("Parameter optimization")
    xgb_model = XGBClassifier(learning_rate=params['learning_rate'],nthread=params['nthread'],n_estimators=params['n_estimators'],silent=params['silent'],objective=params['objective'],booster=params['booster']
        ,gamma=params['gamma'],colsample_bytree=params['colsample_bytree'],subsample=params['subsample'],max_delta_step=params['max_delta_step'],reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],scale_pos_weight=params['scale_pos_weight'],base_score=params['base_score'],seed=params['seed'],
        random_state=params['random_state'],missing=params['missing'],importance_type=params['importance_type'],tree_method=params['tree_method'],num_class=params['num_class'],gpu_id=params['gpu_id']
       )

    clf = GridSearchCV(xgb_model, param_grid=params_grid,verbose=10,cv=5,n_jobs=1,iid=False,scoring='accuracy')
    

    clf.fit(data,label,eval_metric='merror', verbose = True)
    best_est = clf.best_estimator_
    best_params = clf.best_params_
    print("best_est:")
    print(best_est)
    print("best_params:")
    print(best_params)
    print("grid_scores:")
    print(clf.grid_scores_)
    print("best_score:")
    print(clf.best_score_)
    # pkl_name = "best_boston"+"_"
    pickle.dump(clf, open("best_boston_1_8000.pkl", "wb"))

    # Roc AUC with all train data
    prediction_prob = best_est.predict_proba(data)
    prediction = best_est.predict(data)
    print('prediction_prob:')
    print(prediction_prob)
    print('predition:')
    print(prediction)
    prediction_prob_val = best_est.predict_proba(val_data)
    prediction_val = best_est.predict(val_data)
    print('prediction_prob_val:')
    print(prediction_prob_val)
    print('prediction_val:')
    print(prediction_val)
    # print("Roc AUC : ", metrics.roc_auc_score(label, prediction_prob[:,1], average='macro'))
    print("Accuracy: ", metrics.accuracy_score(label, prediction))
    print(confusion_matrix(label,prediction))

    print("val Accuracy: ", metrics.accuracy_score(val_label, prediction_val))
    print(confusion_matrix(val_label,prediction_val))

    print("test")
    probs_test = best_est.predict_proba(test_data)
    label_test = np.argmax(probs_test, axis=1)
    print(label_test.shape)
    print(label_test)

    
    print("test1")
    probs1 = best_est.predict_proba(testb_data)
    label1 = np.argmax(probs, axis=1)
    print(label.shape)
    print(label)

    with open('./result_8000_1_old.csv','w') as f:

        for item in label_test:
            one_hot_label = np.zeros(17)
            one_hot_label[item] = 1
            str_label = str(one_hot_label).replace("[","").replace('.]','').replace('. ',",")+"\n"

            f.write(str_label)
    
    with open('./result_8000_1_new.csv','w') as f1:

        for item in label1:
            one_hot_label = np.zeros(17)
            one_hot_label[item] = 1
            str_label = str(one_hot_label).replace("[","").replace('.]','').replace('. ',",")+"\n"

            f1.write(str_label)

    


    # ================

    # print(clf.best_score_)

    # kf = KFold(n_splits=5, random_state=None, shuffle=True)
    # for train_index, test_index in kf.split(data):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     x_train, x_test = data[train_index], data[test_index]
    #     y_train, y_test = label[train_index], label[test_index]
    #     # model = XGBClassifier(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'],silent=params['silent'],objective=params['objective'],booster=params['booster']
    #     # ,gamma=params['gamma'],subsample=params['subsample'],colsample_bytree=params['colsample_bytree'],reg_alpha=params['reg_alpha'],
    #     # reg_lambda=params['reg_lambda'],scale_pos_weight=params['scale_pos_weight'],base_score=params['base_score'],seed=params['seed'],
    #     # random_state=params['random_state'],missing=params['missing'],importance_type=params['importance_type'],tree_method=params['tree_method'],num_class=params['num_class'],gpu_id=params['gpu_id'])
    #     clf.fit(x_train, y_train,eval_metric='merror', verbose = True, eval_set = [(x_test, y_test)],early_stopping_rounds=100)
    #     print(clf.best_score_)
    #     print(clf.best_params_)
    #     predictions = clf.predict(x_test)
    #     

    # The sklearn API models are picklable

    # must open in binary format to pickle




    # kf = KFold(n_splits=5, random_state=None, shuffle=True)
    # for train_index, test_index in kf.split(data):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     x_train, x_test = data[train_index], data[test_index]
    #     y_train, y_test = label[train_index], label[test_index]
    #     model = XGBClassifier(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'],silent=params['silent'],objective=params['objective'],booster=params['booster']
    #     ,gamma=params['gamma'],subsample=params['subsample'],colsample_bytree=params['colsample_bytree'],reg_alpha=params['reg_alpha'],
    #     reg_lambda=params['reg_lambda'],scale_pos_weight=params['scale_pos_weight'],base_score=params['base_score'],seed=params['seed'],
    #     random_state=params['random_state'],missing=params['missing'],importance_type=params['importance_type'],tree_method=params['tree_method'],num_class=params['num_class'],gpu_id=params['gpu_id'])
    #     model.fit(x_train, y_train,eval_metric='merror', verbose = True, eval_set = [(x_test, y_test)],early_stopping_rounds=100)
    #     predictions = model.predict(x_test)
    #     print(confusion_matrix(y_test,predictions))
    # model.fit()




    
xgboost_classifier()




