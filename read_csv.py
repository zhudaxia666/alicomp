import csv
import numpy as np
from collections import Counter


 #读取csv文件
data1=[]
data2=[]
tem1=[]
tem2=[]
index1=[]
index2=[]
with open('new_test_b_sen2_1.csv','r') as csvfile:
    reader1=csv.reader(csvfile)#读取csv文件，放回的是迭代类型
    for item in reader1:
        index1.append(item.index(max(item)))
        data1.append(item)
    for i in range(len(index1)):
        if index1[i]==0:
            tem1.append(i)
print(tem1)
print(type(data1))
csvfile.close()
with open('new_test_b_sen2_7npy.csv','r') as csvfile:
    reader2=csv.reader(csvfile)#读取csv文件，放回的是迭代类型
    for item in reader2:
        index2.append(item.index(max(item)))
        data2.append(item)
    for i in range(len(index2)):
        if index2[i] == 0:
            tem2.append(i)
print(tem2)
csvfile.close()
index=[]
for i in range(len(data1)):
    if data1[i]!=data2[i]:
        index.append(i+1)

print(len(index))

data=[]
all=[]
data.append(index1)
data.append(index2)
print(np.array(data))
for i in range(len(index2)):
    a=Counter(data[:,i]).most_common(1)
    print(a[0][0])
    break
    # all.append(a[0][0])
# print(np.array(data2).s# hape)
# # print(np.sum(np.array(data2),axis=1))
print(all)
# for i in range(17):
#     sum=0
#     for j in range(len(data2)):
#         sum+=int(data2[j][i])
#         # print(sum)
#     print(sum)

# index=[]
# for i in range(len(data1)):
#     if data1[i]!=data2[i]:
#         index.append(i+1)

# print(len(index))

# sum=np.sum(np.array(data2),axis=0)
# print(sum)
    

