#! /usr/bin/python
import numpy as np
import xgboost as xgb
import h5py
import numpy as np
import os
import operator

base_dir = "data"
path_training = os.path.join(base_dir, "training.h5")
path_training = os.path.join(base_dir, "test.h5")
path_validation = os.path.join(base_dir,"validation.h5/validation.h5")
# path_test = os.path.join(base_dir, "sample_test.h5", "sample_test.h5")
fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')

def Data_change(train=True):
    data=[]
    labels=[]
    if train:
        s1 = fid_training['sen1']
        s2 = fid_training['sen2']
        label = fid_training['label']
    else:
        s1 = fid_validation['sen1']
        s2 = fid_validation['sen2']
        label = fid_validation['label']

    for i in range(1000):
        part=[]
        for chanel1 in range(s1.shape[3]):
            for j1 in range(s1.shape[1]):
                part.extend(s1[i,j1,:,chanel1])
        for chanel2 in range(s2.shape[3]):
            for j2 in range(s2.shape[1]):
                part.extend(s2[i,j2,:,chanel2])
        item=list(label[i])
        labels.append(item.index(1))
        data.append(part)
    print("数据转换成功")
    return np.array(data),np.array(labels)

data,label=Data_change(train=True)
xg_train = xgb.DMatrix(data, label=label)
valid_data,valid_label=Data_change(train=False)
xg_test = xgb.DMatrix(valid_data, label=valid_label)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'   # 多分类的问题
# scale weight of positive examples
param['eta'] = 0.05  # 如同学习率
param['max_depth'] = 12  # 构建树的深度，越大越容易过拟合
param['silent'] = 1  # 设置成1则没有运行信息输出，最好是设置为0
param['nthread'] = 6  # cpu 线程数
param['num_class'] = 17  # 类别数，与 multisoftmax 并用
 
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 10   # 迭代次数
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );
 
print ('predicting, classification error=%f' % (sum( int(pred[i]) != valid_label[i] for i in range(len(valid_label))) / float(len(valid_label)) ))
 
# do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist );
# # Note: this convention has been changed since xgboost-unity
# # get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 6 )
# ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro
 
# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
