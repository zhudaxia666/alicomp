import csv
import numpy as np
import matplotlib.pyplot as plt
csvfile = './result/result_2019-01-08-04:11:55.csv'
data = []
with open(csvfile) as f:
    csv_reader = csv.reader(f)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        data.append(row)
data = [[int(x) for x in row] for row in data] 
data = np.array(data)
print(data.shape)
x = np.sum(data, axis=0)
print(x.shape)
y = [int(i) for i in range(0,17)]

plt.bar(y,x)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()