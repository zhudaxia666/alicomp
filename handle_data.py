# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
import operator

base_dir = "data"
path_training = os.path.join(base_dir, "training.h5")
path_validation = os.path.join(base_dir,"validation.h5/validation.h5")
# path_test = os.path.join(base_dir, "sample_test.h5", "sample_test.h5")
fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')
# print("shape for each channel.")
# s1_training = fid_training['sen1']
# # print(s1_training[0,:,:,:])
# print(len(s1_training.shape))
# print(s1_training.shape)
# s2_training = fid_training['sen2']
# print(s2_training.shape)
# label_training = fid_training['label']
# print(label_training.shape)

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
        labels.append(label[i,:])
        data.append(part)
    print("数据转换成功")
    return data,labels
#     # part=[]
# print(len(label[0]))
# print(len(data[1]))

def Knn_train(data, label):
    # 构建kNN分类器
    neigh = kNN(n_neighbors=10, algorithm='auto', weights='distance', n_jobs=1)
    # 拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(data, label)
    # 验证机数据集
    valid_data,valid_label=Data_change(train=False)
    # 正确检测计数
    resultCount = 0.0
    # 验证数据集的数量
    mTest = len(valid_data)
    valid_data=np.array(valid_data)
    valid_label=np.array(valid_label)
    # classifierResult = neigh.predict(valid_data[1].reshape(1,-1))
    # print((classifierResult==valid_label[1]).all())
    # print(valid_label[1])
    # print(type(valid_label[1]))
    # valid_label=np.array(valid_label)
    for i in range(mTest):
        classifierResult = neigh.predict(valid_data[i].reshape(1,-1))
        # print("分类返回结果为%d\t真实结果为%d" % (classifierResult, valid_label))
        if ((classifierResult==valid_label[i]).all()):
            resultCount += 1.0
    print("总共对了%d个数据\n准确率率为%f%%" % (resultCount, resultCount / mTest * 100))

if __name__ == "__main__":
    data,label=Data_change(train=True)
    Knn_train(data,label)





                

    




