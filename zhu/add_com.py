import numpy as np
from keras.utils import to_categorical
import xgboost as xgb
import h5py
import os
import operator
# from keras.utils import to_categorical
import csv
import random

path_training ="new_train1.h5"
base_dir = "./../data"
# path_training = os.path.join(base_dir, "training.h5")
path_test = os.path.join(base_dir, "test.h5")
# path_validation = os.path.join(base_dir,"validation/validation.h5")

fid_test = h5py.File(path_test,'r')
fid_training = h5py.File(path_training,'r')

def Test_data():
    s1 = fid_test['sen1']
    s2 = fid_test['sen2']
    data=[]
    for i in range(len(s1)):
        part=[]
        for chanel1 in range(s1.shape[3]):
            for j1 in range(s1.shape[1]):
                part.extend(s1[i,j1,:,chanel1])
        for chanel2 in range(s2.shape[3]):
            for j2 in range(s2.shape[1]):
                part.extend(s2[i,j2,:,chanel2])
        data.append(part)
    print("数据转换成功")
    return np.array(data)

def Data_change(train=True):
    data=[]
    labels=[]
    if train:
        s1 = fid_training['sen1']
        s2 = fid_training['sen2']
        label = fid_training['label']
        index=random.sample(range(352366),2000)
    else:
        s1 = fid_validation['sen1']
        s2 = fid_validation['sen2']
        label = fid_validation['label']
        index=random.sample(range(24119),2000)

    for i in index:
        part=[]
        for chanel1 in range(s1.shape[3]):
            for j1 in range(s1.shape[1]):
                part.extend(s1[i,j1,:,chanel1])
        for chanel2 in range(s2.shape[3]):
            for j2 in range(s2.shape[1]):
                part.extend(s2[i,j2,:,chanel2])
        # item=list(label[i])
        labels.append(label[i])
        data.append(part)
    print("数据转换成功")
    return np.array(data),np.array(labels)

def onehot_tranIndex(label):
    index_set=[]
    # label=list(label)
    for i in range(len(label)):
        item=list(label[i])
        index_set.append(item.index(1))
    # write_csvfile(index_set)
    # print(index_set)
    return np.array(index_set)

def write_csvfile(preds):
    one_hots = to_categorical(preds)
    # print(one_hots)
    csvfile=open('csvfile.csv','w',newline='')
    write1=csv.writer(csvfile)
    for i in one_hots:
        write1.writerow(i)
    csvfile.close()


def xgboost_class(train_data,train_label,valid_data,valid_label,test_data):
    params={
    'booster':'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax', 
    'num_class':17, # 类数，与 multisoftmax 并用
    'gamma':0.01,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':6, # 构建树的深度 [1:],典型值：3-10
    #'lambda':450,  # L2 正则项权重
    'subsample':0.5, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1],典型值：0.5-1
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1] 典型值：0.5-1
    #'min_child_weight':12, # 节点的最少特征数
    'silent':1 ,
    'eta': 0.01, # 如同学习率
    'seed':710,
    'nthread':4,# cpu 线程数,根据自己U的个数适当调整
    }

    plst = list(params.items())

    num_rounds = 20 # 迭代你次数
    # xgtest = xgb.DMatrix(test)

    xgtrain = xgb.DMatrix(train_data, label=train_label)
    xgval = xgb.DMatrix(valid_data, label=valid_label)
    xgtest=xgb.DMatrix(test_data)


    # 划分训练集与验证集 
    # xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    # xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    # return 训练和验证的错误率
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]


    # training model 
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
    #model.save_model('./model/xgb.model') # 用于存储训练出的模型
    preds = model.predict(xgtest,ntree_limit=model.best_iteration)
    preds=[int(i) for i in preds]
    return preds


part_data=[]
indexs=[]
for i in range(50):
    train_data,train_label=Data_change(train=True)
    valid_data,valid_label=Data_change(train=False)
    train_label=onehot_tranIndex(train_label)
    valid_label=onehot_tranIndex(valid_label)
    test_data=Test_data()
    item=xgboost_class(train_data,train_label,valid_data,valid_label,test_data) 
    part_data.append(item)
    print("第%d训练完" % (i+1))
data=np.array(part_data)
for i in range(len(part_data[0])):
    counts = np.bincount(data[:,i])
    index=np.argmax(counts)
    indexs.append(index)
indexs=np.array(indexs)
one_hots = to_categorical(indexs).astype(np.int32)
csvfile=open('allaccuracy3.csv','w',newline='')
write1=csv.writer(csvfile)
for i in one_hots:
    write1.writerow(i)
csvfile.close()
print("预测完毕!")






# import numpy as np
# a = np.array([[1,6,3],[1,3,4],[3,0,3]])
# # print(a.shape)
# for i in range(len(a)):
#     counts = np.bincount(a[:,i])
#     print (np.argmax(counts))
#     print(a[:,i])
    
# # counts = np.bincount(a)
# # print (np.argmax(counts))