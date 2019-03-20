# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random

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
def write_h5(sen1,sen2,label,filename):
    # sen1 = np.zeros((shape,32,32,8)) 
    # sen2 = np.zeros((shape,32,32,10))
    # label= np.zeros((shape,17))    
    f = h5py.File(filename,'w')        #创建一个h5文件，文件指针是f  
    f['sen1'] = sen1   
    f['sen2'] = sen2              #将数据写入文件的主键data下面  
    f['label'] = label 
    print("写入完毕")          #将数据写入文件的主键labels下面  
    f.close()                           #关闭文件
def read_h5(filename):
    #HDF5的读取：  
    f = h5py.File(filename,'r')   #打开h5文件  
    # f.keys()
    s1=f['sen1']
    print("s1.shape:",s1.shape)                           #可以查看所有的主键  
    # a = f['data'][:]                    #取出主键为data的所有的键值  
    f.close()  

def tran1(index,train_s1,train_s2,train_label):
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

def tran(index,train_s1,train_s2,train_label):
    sen1=[]
    sen2=[]
    labels=[]
    sub_sen1 = np.zeros((32,32,8)) 
    sub_sen2 = np.zeros((32,32,10))
    for i in index:
        #将数据左右旋转
        for j in range(8):
            sub_sen1[:,:,j] = train_s1[i][:,:,j]
        for k in range(10):
            sub_sen2[:,:,k] = train_s2[i][:,:,k]
        # sub_label=label[i]
        sen1.append(sub_sen1)
        sen2.append(sub_sen2)
        labels.append(train_label[i])
    return sen1,sen2,labels

base_dir = "./../data"
path_training = os.path.join(base_dir, "training.h5")
# path_test = os.path.join(base_dir, "test.h5")
path_validation = os.path.join(base_dir,"validation.h5/validation.h5")

fid_training = h5py.File(path_training,'r')
fid_val = h5py.File(path_validation,'r')

train_s1=fid_training['sen1']
train_s2=fid_training['sen2']
train_label=fid_training['label']

val_s1=fid_val['sen1']
# print(val_s1.shape)
val_s2=fid_val['sen2']
val_label=fid_val['label']

# print(val_s1.shape)

index1=[]
index2=[]
index3=[]
index4=[]
index5=[]
index6=[]
index7=[]
index8=[]
index9=[]
index10=[]
index11=[]
index12=[]
index13=[]
index14=[]
index15=[]
index16=[]
index17=[]
index18=[]
index19=[]
index20=[]

data=[]
s1=[]
s2=[]
label=[]
# indexa=[1,2,3,4,5,7,8,9,10,11,12,13,15,16]
for i in range(len(train_label)):
    if (list(train_label[i]).index(1)==1):
        index1.append(i)
    if (list(train_label[i]).index(1)==2):
        index2.append(i)
    if (list(train_label[i]).index(1)==3):
        index3.append(i)
    if (list(train_label[i]).index(1)==4):
        index4.append(i)
    if (list(train_label[i]).index(1)==5):
        index5.append(i)
    if (list(train_label[i]).index(1)==7):
        index6.append(i)
    if (list(train_label[i]).index(1)==8):
        index7.append(i)
    if (list(train_label[i]).index(1)==9):
        index8.append(i)
    if (list(train_label[i]).index(1)==10):
        index9.append(i)
    if (list(train_label[i]).index(1)==11):
        index10.append(i)
    if (list(train_label[i]).index(1)==12):
        index11.append(i)
    if (list(train_label[i]).index(1)==13):
        index12.append(i)
    if (list(train_label[i]).index(1)==15):
        index13.append(i)
    if (list(train_label[i]).index(1)==16):
        index14.append(i)
    if (list(train_label[i]).index(1)==0):
        index15.append(i)
    if (list(train_label[i]).index(1)==6):
        index16.append(i)
    if (list(train_label[i]).index(1)==14):
        index17.append(i)

index18=random.sample(index15,4950)
index19=random.sample(index16,3140)
index20=random.sample(index17,2360)

index15=random.sample(index15,700)
index16=random.sample(index16,1100)
index17=random.sample(index17,1360)


#   [ 6746. 5647. 7151. 7243. 6094.  4605. 6086. 7140. 5713. 7618.
#  6798. 5253.  7328. 5391.]  
index1=random.sample(index1,6750)
index2=random.sample(index2,5650)
index3=random.sample(index3,7160)
index4=random.sample(index4,7243)
index5=random.sample(index5,6100)
index6=random.sample(index6,4610)
index7=random.sample(index7,6090)
index8=random.sample(index8,7140)
index9=random.sample(index9,5720)
index10=random.sample(index10,7620)
index11=random.sample(index11,6800)
index12=random.sample(index12,5260)
index13=random.sample(index13,7330)
index14=random.sample(index14,5391)


sen11,sen21,label1=tran(index1,train_s1,train_s2,train_label)
sen12,sen22,label2=tran(index2,train_s1,train_s2,train_label)
sen13,sen23,label3=tran(index3,train_s1,train_s2,train_label)
sen14,sen24,label4=tran(index4,train_s1,train_s2,train_label)
sen15,sen25,label5=tran(index5,train_s1,train_s2,train_label)
sen16,sen26,label6=tran(index6,train_s1,train_s2,train_label)
sen17,sen27,label7=tran(index7,train_s1,train_s2,train_label)
sen18,sen28,label8=tran(index8,train_s1,train_s2,train_label)
sen19,sen29,label9=tran(index9,train_s1,train_s2,train_label)
sen110,sen210,label10=tran(index10,train_s1,train_s2,train_label)
sen111,sen211,label11=tran(index11,train_s1,train_s2,train_label)
sen112,sen212,label12=tran(index12,train_s1,train_s2,train_label)
sen113,sen213,label13=tran(index13,train_s1,train_s2,train_label)
sen114,sen214,label14=tran(index14,train_s1,train_s2,train_label)
sen115,sen215,label15=tran1(index15,train_s1,train_s2,train_label)
sen116,sen216,label16=tran1(index16,train_s1,train_s2,train_label)
sen117,sen217,label17=tran1(index17,train_s1,train_s2,train_label)
sen118,sen218,label18=tran(index18,train_s1,train_s2,train_label)
sen119,sen219,label19=tran(index19,train_s1,train_s2,train_label)
sen120,sen220,label20=tran(index20,train_s1,train_s2,train_label)

s1.extend(sen11)
s1.extend(sen12)
s1.extend(sen13)
s1.extend(sen14)
s1.extend(sen15)
s1.extend(sen16)
s1.extend(sen17)
s1.extend(sen18)
s1.extend(sen19)
s1.extend(sen110)
s1.extend(sen111)
s1.extend(sen112)
s1.extend(sen113)
s1.extend(sen114)
s1.extend(sen115)
s1.extend(sen116)
s1.extend(sen117)
s1.extend(sen118)
s1.extend(sen119)
s1.extend(sen120)
s1.extend(val_s1)
# print(np.array(s1).shape)

# s1.extend(val_s1)
# s1.extend(train_s1)

s2.extend(sen21)
s2.extend(sen22)
s2.extend(sen23)
s2.extend(sen24)
s2.extend(sen25)
s2.extend(sen26)
s2.extend(sen27)
s2.extend(sen28)
s2.extend(sen29)
s2.extend(sen210)
s2.extend(sen211)
s2.extend(sen212)
s2.extend(sen213)
s2.extend(sen214)
s2.extend(sen215)
s2.extend(sen216)
s2.extend(sen217)
s2.extend(sen218)
s2.extend(sen219)
s2.extend(sen220)
s2.extend(val_s2)
# print(np.array(s2).shape)

# s2.extend(val_s2)
# s2.extend(train_s2)

label.extend(label1)
label.extend(label2)
label.extend(label3)
label.extend(label4)
label.extend(label5)
label.extend(label6)
label.extend(label7)
label.extend(label8)
label.extend(label9)
label.extend(label10)
label.extend(label11)
label.extend(label12)
label.extend(label13)
label.extend(label14)
label.extend(label15)
label.extend(label16)
label.extend(label17)
label.extend(label18)
label.extend(label19)
label.extend(label20)
label.extend(val_label)
# print(np.array(label).shape)

# label.extend(val_label)

# print(np.array(s1).shape)
# print(np.array(s2).shape)
# print(np.array(label).shape)

# a.extend(s1)
print("添加进去了·")
filename='new_train1.h5'
write_h5(s1,s2,label,filename)
