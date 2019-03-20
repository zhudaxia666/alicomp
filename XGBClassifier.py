import numpy as np
from xgboost import XGBClassifier
import h5py
import os
import readData

base_dir='data'
#new_train1.h5数据集里每一类标签都是8千个数据
ft_train_path=os.path.join(base_dir,"new_train1.h5")
ft_train=h5py.File(ft_train_path,'r')

new_train1_sen1=ft_train['sen1']
print(new_train1_sen1.shape)
new_train1_sen2=ft_train['sen2']
new_train1_label=ft_train['label']
a=np.sum(new_train1_label,0)
print(a)

# new_train1_X=
# new_train1_Y=
# my_model=XGBClassifier()
# my_model.fit(new_train1_sen2,new_train1_label,verbose=False)
# predictions = my_model.predict(test_X)

