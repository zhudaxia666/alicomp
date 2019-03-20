import h5py
import numpy as np
import random
import gc
import xgboost as xgb
from keras.utils import to_categorical
import csv
import os

base_dir = "data"
# path_training = os.path.join(base_dir, "training.h5")
path_test = os.path.join(base_dir, "test_b.h5")
path_train = os.path.join(base_dir, "validation.h5")
# path_validation = os.path.join(base_dir,"validation/validation.h5")
# fid_train = h5py.File(path_train,'r')
fid_training = h5py.File(path_test, 'r')


# fid_val = h5py.File(path_validation,'r')
# test_s1=fid_training['sen1']
# test_s2=fid_training['sen2']
# train_label=fid_training['label']
def Test_data():
#将sen2的数据按照2轴
    # s1 = fid_training['sen1']
    s2 = fid_training['sen2']
    data = []
    for i in range(len(s2)):
        part = []
        # for chanel1 in range(s1.shape[3]):
        #     for j1 in range(s1.shape[1]):
        #         part.extend(s1[i, j1, :, chanel1])
        for chanel2 in range(s2.shape[3]):
            for j2 in range(s2.shape[1]):
                part.extend(s2[i, j2, :, chanel2])
        data.append(part)
    print("数据转换成功")
    return np.array(data)


def get_all_data(counts,tr_s2):
    # s1_data = tr_s1
    s2_data = tr_s2

    res_list = []
    for item in range(counts):
        per_item = []
        # this_item_1 = s1_data[int(item)]
        this_item_2 = s2_data[int(item)]
        # for num in range(0, 8):
        #     per_item.append(this_item_1[:, :, num])
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


def write_h5(sen2, label, filename):
    # sen1 = np.zeros((shape,32,32,8))
    # sen2 = np.zeros((shape,32,32,10))
    # label= np.zeros((shape,17))
    f = h5py.File(filename, 'w')  # 创建一个h5文件，文件指针是f
    # f['sen1'] = sen1
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


def xgboost_class(train_data, train_label, valid_data, valid_label, test_data):
    params = {
        'booster': 'gbtree',
        # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
        'objective': 'multi:softmax',
        'num_class': 17,  # 类数，与 multisoftmax 并用
        'gamma': 0.01,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
        'max_depth': 7,  # 构建树的深度 [1:],典型值：3-10
        # 'lambda':450,  # L2 正则项权重
        'subsample': 0.5,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1],典型值：0.5-1
        'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1] 典型值：0.5-1
        # 'min_child_weight':12, # 节点的最少特征数
        'silent': 1,
        'eta': 0.32,  # 如同学习率0.3准确率0.63多
        'seed': 710,
        'nthread': 4,  # cpu 线程数,根据自己U的个数适当调整
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


def result(train_sen2, train_label):
    # tr_s1 = []
    tr_s2 = []
    tr_label = []

    # te_s1 = []
    te_s2 = []
    te_label = []

    train = random.sample(range(len(train_label)), 20000)
    test = list(set(range(len(train_label))).difference(set(train)))
    # test=list(np.delete(range(125623),train))
    test = random.sample(test, 4000)

    for i in train:
        # tr_s1.append(train_sen1[i])
        tr_s2.append(train_sen2[i])
        tr_label.append(train_label[i])

    for k in test:
        # te_s1.append(train_sen1[k])
        te_s2.append(train_sen2[k])
        te_label.append(train_label[k])

    train_data = get_all_data(20000,tr_s2).reshape(20000, -1)
    # print(train_data.shape) #(8000, 18, 32, 32)
    val_data = get_all_data(4000,te_s2).reshape(4000, -1)

    train_label = get_label(20000, tr_label)
    train_label = onehot_tranIndex(train_label)
    val_label = get_label(4000, te_label)
    val_label = onehot_tranIndex(val_label)

    test_data = Test_data()

    item = xgboost_class(train_data, train_label, val_data, val_label, test_data)
    print("预测完毕!")
    return item


file = h5py.File(path_train, 'r')
# train_sen1 = file["sen1"]
train_sen2 = file["sen2"]
train_label = file["label"]

part_data = []
indexs = []

for i in range(20):
    item = result(train_sen2, train_label)
    part_data.append(item)
    print("第%d训练完" % (i + 1))

data = np.array(part_data)
for i in range(len(part_data[0])):
    counts = np.bincount(data[:, i])
    index = np.argmax(counts)
    indexs.append(index)
indexs = np.array(indexs)
one_hots = to_categorical(indexs).astype(np.int32)
csvfile = open('new_accuracy_test_b.csv', 'w', newline='')
write1 = csv.writer(csvfile)
for i in one_hots:
    write1.writerow(i)
csvfile.close()
print("预测完毕!")
