import os
import h5py
import numpy as np
import random
import gc

def zhuanhuan(index,train_s1,train_s2,train_label):
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
    # print(np.array(sen1).shape)
    # print(np.array(sen2).shape)
    # print(np.array(labels).shape)
    return sen1,sen2,labels
   

base_dir = "./../data"
path_training = os.path.join(base_dir, "training.h5")
# path_test = os.path.join(base_dir, "test.h5")
# path_validation = os.path.join(base_dir,"validation/validation.h5")

fid_training = h5py.File(path_training,'r')
# fid_val = h5py.File(path_validation,'r')
train_s1=fid_training['sen1']
train_s2=fid_training['sen2']
train_label=fid_training['label']
#读取所需要的数据

#随机选取8000个数据
indexa=[1,2,3,4,5,7,8,9,10,11,12,13,15,16]
index_all=[]
#找出对应label的个数
# del train_s1,train_s2
# gc.collect()
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

# for i in range(len(index_all)):
#     print(len(index_all[i]))

for i in range(len(index_all)):
    s1=[]
    s2=[]
    label=[]
    item1=random.sample(index_all[i],7800)
    # a,b,c=tran(item1,train_s1,train_s2,train_label)
    # #写数据
    sen1,sen2,label1=zhuanhuan(item1,train_s1,train_s2,train_label)
    s1.extend(sen1)
    s2.extend(sen2)
    label.extend(label1)
    i_str=str(indexa[i])
    filename=i_str+'.h5'
    f=h5py.File(filename,'w')
    # file.create_dataset("sen1",data=s1)
    # file.create_dataset("sen2",data=s2)
    # file.create_dataset("label",data=label)
    f['sen1'] = np.array(s1)   
    f['sen2'] = np.array(s2)              #将数据写入文件的主键data下面  
    f['label'] = np.array(label)
    print("写入完毕！")
    f.close()
    del s1,s2,label,sen1,sen2,label1
    gc.collect()
    break