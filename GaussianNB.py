# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNN
import operator

base_dir = "data"
path_training = os.path.join(base_dir, "training.h5")
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

def Naive_train(data, label):
    clf=GaussianNB().fit(data,label)
    # 验证机数据集
    valid_data,valid_label=Data_change(train=False)
    # 正确检测计数
    resultCount = 0.0
    # 验证数据集的数量
    mTest = len(valid_data)
    
    for i in range(mTest):
        classifierResult = clf.predict(valid_data[i].reshape(1,-1))
        if (classifierResult==valid_label[i]):
            resultCount += 1.0
    print("总共对了%d个数据\n准确率率为%f%%" % (resultCount, resultCount / mTest * 100))

if __name__ == "__main__":
    data,label=Data_change(train=True)
    Naive_train(data,label)





                

    




