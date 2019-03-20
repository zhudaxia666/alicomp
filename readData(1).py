# 查看当前kernel下已安装的包  list packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt




class readData(object):
    def __init__(self):
        self.base_dir = "data"
        self.path_training = os.path.join(self.base_dir, "training.h5")
        self.path_validation = os.path.join(self.base_dir,"validation.h5/validation.h5")
        self.fid_training = h5py.File(self.path_training,'r')
        self.fid_validation = h5py.File(self.path_validation,'r')

    def get_sen1_data(self, bias, img_list=[], counts=1, train=True):
        s1_data = None
        if train:
            s1_data = self.fid_training['sen1']
        else:
            s1_data = self.fid_validation['sen1']

        if len(img_list)!=0:
            res_list = []
            for item in range(counts):
                per_item = []
                this_item = s1_data[int(bias)+int(item)]
                for num in img_list:
                    per_item.append(this_item[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat

        else:
                
            res_list = []
            for item in range(counts):
                per_item = []
                this_item = s1_data[int(bias)+int(item)]
                for num in range(0,8):
                    per_item.append(this_item[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat

    def get_sen2_data(self, bias, img_list=[], counts=1, train=True):
        s2_data = None
        if train:
            s2_data = self.fid_training['sen2']
        else:
            s2_data = self.fid_validation['sen2']     
        if len(img_list)!=0:
            res_list = []
            for item in range(counts):
                per_item = []
                this_item = s2_data[int(bias)+int(item)]
                for num in img_list:
                    per_item.append(this_item[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat

        else:
                
            res_list = []
            for item in range(counts):
                per_item = []
                this_item = s2_data[int(bias)+int(item)]
                for num in range(0,10):
                    per_item.append(this_item[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat

    def get_all_data(self, bias, sen1_list=[], sen2_list=[], counts=1, train=True):
        s1_data = None
        s2_data = None
        if train:
            s1_data = self.fid_training['sen1']
            s2_data = self.fid_training['sen2']
        else:
            s1_data = self.fid_validation['sen1']  
            s2_data = self.fid_validation['sen2'] 
        if len(sen1_list)!=0 and len(sen2_list) != 0:  
            res_list = []
            for item in range(counts):
                per_item = []
                this_item_1 = s1_data[int(bias)+int(item)]
                this_item_2 = s2_data[int(bias)+int(item)]
                for num in sen1_list:
                    per_item.append(this_item_1[:,:,num])
                for num in sen2_list:
                    per_item.append(this_item_2[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat

        elif len(sen1_list)!=0 and len(sen2_list) == 0:             
            res_list = []
            for item in range(counts):
                per_item = []
                this_item_1 = s1_data[int(bias)+int(item)]
                this_item_2 = s2_data[int(bias)+int(item)]
                for num in sen1_list:
                    per_item.append(this_item_1[:,:,num])
                for num in range(0,10):
                    per_item.append(this_item_2[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat
        elif len(sen1_list)==0 and len(sen2_list) != 0:             
            res_list = []
            for item in range(counts):
                per_item = []
                this_item_1 = s1_data[int(bias)+int(item)]
                this_item_2 = s2_data[int(bias)+int(item)]
                for num in (0,7):
                    per_item.append(this_item_1[:,:,num])
                for num in sen2_list:
                    per_item.append(this_item_2[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat
        else:
            res_list = []
            for item in range(counts):
                per_item = []
                this_item_1 = s1_data[int(bias)+int(item)]
                this_item_2 = s2_data[int(bias)+int(item)]
                for num in range(0,8):
                    per_item.append(this_item_1[:,:,num])
                for num in range(0,10):
                    per_item.append(this_item_2[:,:,num])
                res_list.append(np.array(per_item))
            res_mat = np.array(res_list)
            return res_mat
        


    def get_label(self, bias, counts=1, train=True):
        label = None
        if train:
            label = self.fid_training['label']
        else:
            label = self.fid_training['label']   
        res_list = []
        for item in range(counts):
            per_item = []
            this_item = label[int(bias)+int(item)]
            res_list.append(this_item)
        res_mat = np.array(res_list)
        return res_mat


    def get_length(self, train=True):
        if train:
            return self.fid_training['label'].shape[0]
        else:
            return self.fid_validation['label'].shape[0]
a = readData()
a1 = a.get_all_data(12908,[2,3],[4,5],2)
print(a1.reshape(1,-1).shape)

b = a.get_length(False)
print(b)


