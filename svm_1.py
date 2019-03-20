# -*- coding: UTF-8 -*-
from sklearn.svm import SVC
import h5py
import numpy as np
import random
import gc
# import xgboost as xgb
from keras.utils import to_categorical
import csv
import os

base_dir = "data"
# base_dir2 = "zhu"
# path_training = os.path.join(base_dir, "training.h5")
path_test = os.path.join(base_dir, "test.h5")
path_train = os.path.join(base_dir, "new_train1.h5")
# path_validation = os.path.join(base_dir,"validation/validation.h5")
# fid_train = h5py.File(path_train,'r')
fid_training = h5py.File(path_test, 'r')


# fid_val = h5py.File(path_validation,'r')
# test_s1=fid_training['sen1']
# test_s2=fid_training['sen2']
# train_label=fid_training['label']
def Test_data():
    s1 = fid_training['sen1']
    s2 = fid_training['sen2']
    data = []
    for i in range(len(s1)):
        part = []
        for chanel1 in range(s1.shape[3]):
            for j1 in range(s1.shape[1]):
                part.extend(s1[i, j1, :, chanel1])
        for chanel2 in range(s2.shape[3]):
            for j2 in range(s2.shape[1]):
                part.extend(s2[i, j2, :, chanel2])
        data.append(part)
    print("数据转换成功")
    return np.array(data)


def get_all_data(counts, tr_s1, tr_s2):
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


def write_h5(sen1, sen2, label, filename):
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


def handwritingClassTest(train_data, train_label, test_data):
    # # 测试集的Labels
    # hwLabels = []
    # # 返回trainingDigits目录下的文件名
    # trainingFileList = listdir('trainingDigits')
    # # 返回文件夹下文件的个数
    # m = len(trainingFileList)
    # # 初始化训练的Mat矩阵,测试集
    # trainingMat = np.zeros((m, 1024))
    # # 从文件名中解析出训练集的类别
    # for i in range(m):
    #     # 获得文件的名字
    #     fileNameStr = trainingFileList[i]
    #     # 获得分类的数字
    #     classNumber = int(fileNameStr.split('_')[0])
    #     # 将获得的类别添加到hwLabels中
    #     hwLabels.append(classNumber)
    #     # 将每一个文件的1x1024数据存储到trainingMat矩阵中
    #     trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # kernel 算法中采用的核函数类型，可选参数有：
    # ‘linear’:线性核函数
    # ‘poly’：多项式核函数
    # ‘rbf’：径像核函数/高斯核
    # ‘sigmod’:sigmod核函数
    # ‘precomputed’:核矩阵
    # C 错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。
    # 对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
    clf = SVC(C=200, kernel='rbf')
    clf.fit(train_data, train_label)
    # # 返回testDigits目录下的文件列表
    # testFileList = listdir('testDigits')
    # # 错误检测计数
    # errorCount = 0.0
    # 测试数据的数量
    mTest = len(test_data)
    test_label = []
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # # 获得文件的名字
        # fileNameStr = testFileList[i]
        # # 获得分类的数字
        # classNumber = int(fileNameStr.split('_')[0])
        # # 获得测试集的1x1024向量,用于训练
        # vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = clf.predict(test_data[i].reshape(1, -1))
        test_label.append(classifierResult)
        # print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        # if (classifierResult != classNumber):
        #     errorCount += 1.0
    # print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))
    return test_label


def result(train_sen1, train_sen2, train_label):
    tr_s1 = []
    tr_s2 = []
    tr_label = []

    train = random.sample(range(136073), 10000)
    # test = list(set(range(136073)).difference(set(train)))
    # # test=list(np.delete(range(125623),train))
    # test = random.sample(test, 8000)

    for i in train:
        tr_s1.append(train_sen1[i])
        tr_s2.append(train_sen2[i])
        tr_label.append(train_label[i])

    # for k in test:
    #     te_s1.append(train_sen1[k])
    #     te_s2.append(train_sen2[k])
    #     te_label.append(train_label[k])

    train_data = get_all_data(10000, tr_s1, tr_s2).reshape(10000, -1)
    # print(train_data.shape) #(8000, 18, 32, 32)
    # val_data = get_all_data(8000, te_s1, te_s2).reshape(8000, -1)

    train_label = get_label(10000, tr_label)
    train_label = onehot_tranIndex(train_label)
    # val_label = get_label(8000, te_label)
    # val_label = onehot_tranIndex(val_label)

    test_data = Test_data()

    item = handwritingClassTest(train_data, train_label, test_data)
    print("预测完毕!")
    print(len(item))
    return item


file = h5py.File(path_train, 'r')
train_sen1 = file["sen1"]
train_sen2 = file["sen2"]
train_label = file["label"]

part_data = []
indexs = []

for i in range(10):
    item=result(train_sen1,train_sen2,train_label)
    part_data.append(item)
    print("第%d训练完" % (i+1))
# item = result(train_sen1, train_sen2, train_label)
# part_data.append(item)
# print("第%d训练完" % (1))
data=np.array(part_data)
for i in range(len(part_data[0])):
    counts = np.bincount(data[:,i])
    index=np.argmax(counts)
    indexs.append(index)
indexs = np.array(indexs)
one_hots = to_categorical(indexs).astype(np.int32)
csvfile = open('svm_accuracy.csv', 'w', newline='')
write1 = csv.writer(csvfile)
for i in one_hots:
    write1.writerow(i)
csvfile.close()
print("预测完毕!")

# if __name__ == '__main__':
#     handwritingClassTest(train_data,train_label,test_data)
