# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gc

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
def write_h5(shape,sen1,sen2,label,filename):
    sen1 = np.zeros((shape,32,32,8)) 
    sen2 = np.zeros((shape,32,32,10))
    label= np.zeros((shape,17))    
    f = h5py.File(filename,'w')        #创建一个h5文件，文件指针是f  
    f['sen1'] = sen1   
    f['sen2'] = sen2              #将数据写入文件的主键data下面  
    f['label'] = label 
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
def zhuanhuan(train_s1,train_s2,train_label,val_s1,val_s2,val_label,filename):
    # dataset=[]
    sen1=[]
    sen2=[]
    labels=[]
    sub_sen1 = np.zeros((32,32,8)) 
    sub_sen2 = np.zeros((32,32,10))

    for i in range(len(train_label)):
        for j in range(8):
            sub_sen1[:,:,j] = train_s1[i][:,:,j]
        for k in range(10):
            sub_sen2[:,:,k] = train_s2[i][:,:,k]
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])

    for i in range(len(val_label)):
        for j in range(8):
            sub_sen1[:,:,j] = val_s1[i][:,:,j]
        for k in range(10):
            sub_sen2[:,:,k] = val_s2[i][:,:,k]
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(val_label[i])
    # print(np.array(sen1).shape)
    # print(np.array(sen2).shape)
    # print(np.array(labels).shape)
    write_h5((len(val_label)+len(train_label)),sen1,sen2,labels,filename)
    read_h5(filename)
def tran(index,train_s1,train_s2,train_label):
    sen1=[]
    sen2=[]
    labels=[]
    sub_sen1 = np.zeros((32,32,8)) 
    sub_sen2 = np.zeros((32,32,10))
    for i in index:
        #将数据左右旋转
        for j in range(8):
            sub_sen1[:,:,j] = FZ2(train_s1[i][:,:,j])
        for k in range(10):
            sub_sen2[:,:,k] = FZ2(train_s2[i][:,:,k])
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])
        ##将数据顺时针90旋转
        for j in range(8):
            sub_sen1[:,:,j] = FZ4(train_s1[i][:,:,j])
        for k in range(10):
            sub_sen2[:,:,k] = FZ4(train_s2[i][:,:,k])
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])
        ##将数据逆时针90旋转
        for j in range(8):
            sub_sen1[:,:,j] = FZ5(train_s1[i][:,:,j])
        for k in range(10):
            sub_sen2[:,:,k] = FZ5(train_s2[i][:,:,k])
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])
        ##将数据180旋转
        for j in range(8):
            sub_sen1[:,:,j] = FZ1(train_s1[i][:,:,j])
        for k in range(10):
            sub_sen2[:,:,k] = FZ1(train_s2[i][:,:,k])
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])
    return sen1,sen2,labels
def chouqu(label,j):
    index=[]
    for i in range(len(label)):
        if (list(label[i]).index(1)==j):
            index.append(i)
    return index


base_dir = "./../data"
path_training = os.path.join(base_dir, "training.h5")
# path_test = os.path.join(base_dir, "test.h5")
path_validation = os.path.join(base_dir,"validation.h5/validation.h5")

fid_training = h5py.File(path_training,'r')
fid_val = h5py.File(path_validation,'r')

train_s1=fid_training['sen1']
train_s2=fid_training['sen2']
train_label=fid_training['label']

# val_s1=fid_val['sen1']
# # print(val_s1.shape)
# val_s2=fid_val['sen2']
# val_label=fid_val['label']

# print(val_s1.shape)

index1=[]
index2=[]
index3=[]

data=[]
s1=[]
s2=[]
label=[]

# for i in range(len(train_label)):
#     if (list(train_label[i]).index(1)==0):
#         index1.append(i)
#     if (list(train_label[i]).index(1)==6):
#         index2.append(i)
#     if (list(train_label[i]).index(1)==14):
#         index3.append(i)
# index1=random.sample(index1,700)
# index2=random.sample(index2,1100)
# index3=random.sample(index3,1360)

#随机选取8000个数据
indexa=[1,2,3,4,5,7,8,9,10,11,12,13,15,16]
index_all=[]
#找出对应label的个数
del train_s1,train_s2
gc.collect()
for i in indexa:
    item=[]
    for j in range(len(train_label)):
        if (list(train_label[j]).index(1)==i):
            item.append(j)
            # del train_s1,train_s2
            # gc.collect()
    # del train_s1,train_s2
    # gc.collect()
    index_all.append(item)
    # del train_s1,train_s2
    # gc.collect()

# print(len(index_all))#14
# print(len(index_all[0]))#24431
train_s1=fid_training['sen1']
train_s2=fid_training['sen2']
# train_label=fid_training['label']
for i in range(len(index_all)):
    item1=random.sample(index_all[i],7800)
    # a,b,c=tran(item1,train_s1,train_s2,train_label)
    for j in item1:
        s1.append(train_s1[j])
        s2.append(train_s2[j])
        label.append(train_label[j])
    del item1
    gc.collect()
        

del train_s1,train_s2,train_label
gc.collect()
print(len(s1))

# sen11,sen21,label1=tran(index1,train_s1,train_s2,train_label)
# sen12,sen22,label2=tran(index2,train_s1,train_s2,train_label)
# sen13,sen23,label3=tran(index3,train_s1,train_s2,train_label)

# s1.extend(sen11)
# s1.extend(sen12)
# s1.extend(sen13)
# print(np.array(s1).shape)

# s1.extend(val_s1)
# # s1.extend(train_s1)

# s2.extend(sen21)
# s2.extend(sen22)
# s2.extend(sen23)
# print(np.array(s2).shape)

# s2.extend(val_s2)
# # s2.extend(train_s2)

# label.extend(label1)
# label.extend(label2)
# label.extend(label3)
# print(np.array(label).shape)

# label.extend(val_label)

# print(np.array(s1).shape)
# print(np.array(s2).shape)
# print(np.array(label).shape)

# # a.extend(s1)

# filename='new_val.h5'
# write_h5(len(s1),s1,s2,label,filename)
