# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
# from sklearn.naive_bayes import GaussianNB
# from readData import *
#上下旋转
# A[::-1]这个操作对于行向量可以左右翻转；对于二维矩阵可以实现上下翻转
def fz(a):
    return a[::-1]
#旋转180度
def FZ1(mat):
    return np.array(fz(list(map(fz, mat))))
#左右旋转
def FZ2(mat):
    return np.array(list(map(fz, mat)))
#转置
def FZ3(mat):
    return np.array(list(map(list,zip(*mat))))
#顺时针旋转90
def FZ4(mat):
    return np.array(FZ3(fz(mat)))                                                                                                                                                            
#逆时针旋转90
def FZ5(mat):
    return np.array(FZ3(FZ2(mat)))
def write_h5(shape,filename):
    # sen1 = np.zeros((shape,32,32,8))
    # sen2 = np.zeros((shape,32,32,10))
    # label= np.zeros((shape,17))
    f = h5py.File(filename,'w')        #创建一个h5文件，文件指针是f  
    f['sen1'] = np.zeros((shape,32,32,8))
    f['sen2'] = np.zeros((shape,32,32,10))              #将数据写入文件的主键data下面
    f['label'] = np.zeros((shape,17))
    # print("")          #将数据写入文件的主键labels下面  
    f.close()                           #关闭文件
def read_h5(filename):
    #HDF5的读取：  
    f = h5py.File(filename,'r')   #打开h5文件  
    # f.keys()
    s1=f['sen1']
    print("s1.shape:",s1.shape)                           #可以查看所有的主键  
    # a = f['data'][:]                    #取出主键为data的所有的键值  
    f.close()  
def zhuanhuan(s1,s2,label,indexs,filename):
    # dataset=[]
    sen1=[]
    sen2=[]
    labels=[]
    sub_sen1 = np.zeros((32,32,8)) 
    sub_sen2 = np.zeros((32,32,10))
    for i in range(len(label)):
        if (list(label[i]).index(1) in indexs):
            #将数据左右旋转
            for j in range(8):
                sub_sen1[:,:,j] = FZ2(s1[i][:,:,j])
            for k in range(10):
                sub_sen2[:,:,k] = FZ2(s2[i][:,:,k])
            # sub_label=label[i]
            sen1.append(sub_sen1)
            sen2.append(sub_sen2)
            labels.append(label[i])
            ##将数据顺时针90旋转
            for j in range(8):
                sub_sen1[:,:,j] = FZ4(s1[i][:,:,j])
            for k in range(10):
                sub_sen2[:,:,k] = FZ4(s2[i][:,:,k])
            # sub_label=label[i]
            sen1.append(sub_sen1)
            sen2.append(sub_sen2)
            labels.append(label[i])
            ##将数据逆时针90旋转
            for j in range(8):
                sub_sen1[:,:,j] = FZ5(s1[i][:,:,j])
            for k in range(10):
                sub_sen2[:,:,k] = FZ5(s2[i][:,:,k])
            # sub_label=label[i]
            sen1.append(sub_sen1)
            sen2.append(sub_sen2)
            labels.append(label[i])
            ##将数据180旋转
            for j in range(8):
                sub_sen1[:,:,j] = FZ1(s1[i][:,:,j])
            for k in range(10):
                sub_sen2[:,:,k] = FZ1(s2[i][:,:,k])
            # sub_label=label[i]
            sen1.append(sub_sen1)
            sen2.append(sub_sen2)
            labels.append(label[i])
    print(np.array(sen1).shape)
    print(np.array(sen2).shape)
    print(np.array(labels).shape)
    write_h5(np.array(sen1).shape[0],sen1,sen2,labels,filename)

base_dir = "./../data"
path_training = os.path.join(base_dir, "training.h5")
path_test = os.path.join(base_dir, "test.h5")
path_validation = os.path.join(base_dir,"validation.h5")

fid_training = h5py.File(path_training,'r')
fid_val = h5py.File(path_validation,'r')

train_s1=fid_training['sen1']
train_s2=fid_training['sen2']
train_label=fid_training['label']

val_s1=fid_val['sen1']
val_s2=fid_val['sen2']
val_label=fid_val['label']

train_index=[0,6,14]
val_index=[0,6,11,14,15]
train_filename='train.h5'
val_filename="val.h5"

zhuanhuan(train_s1,train_s2,train_label,train_index,train_filename)
zhuanhuan(val_s1,val_s2,val_label,val_index,val_filename)
