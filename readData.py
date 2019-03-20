# 查看当前kernel下已安装的包  list packages
import os

import cv2
import h5py
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
from torch.nn import functional as F
import torch as t

class readData(object):


    def __init__(self):
        # self.base_dir = "../data"
        # self.path_training = "./data/train/new_training8000_1.h5"
        # self.path_validation = "./data/val/new_validation8000_1.h5"
        # self.path_test = "./data/test/round1_test_a_20181109.h5"
        # self.path_training = "./data/train/new_training8000_1.h5"
        # self.path_validation = "./data/val/new_validation8000_1.h5"
        self.path_training = "./data/new_train8000.h5"
        self.path_validation = "./data/validation.h5"
        self.path_test = "./data/round1_test_a_20181109.h5"
        self.path_original_training='./data/new_train8000.h5'
        self.path_newTrain='./data/label_0_6_11_14_train.h5'
        self.path_test_2 = "./data/round1_test_b_20190104.h5"
        self.fid_training = h5py.File(self.path_training,'r')
        self.fid_validation = h5py.File(self.path_validation,'r')
        self.fid_test = h5py.File(self.path_test, "r")
        self.fid_newTrain = h5py.File(self.path_newTrain, 'w')
        self.fid_original_training = h5py.File(self.path_original_training, 'r')
        self.fid_test_2 = h5py.File(self.path_test_2,"r")
    def generateDataset(self,indexList=[]):
        orig_sen1 = self.fid_original_training['sen1']
        orig_sen2 = self.fid_original_training['sen2']
        orig_label = self.fid_original_training['label']
        # fid_newTrain_sen1 = self.fid_newTrain.create_dataset('sen1', shape=(1, 32, 32, 8), maxshape=(None, 32, 32, 8), chunks=(1, 32, 32, 8))
        # fid_newTrain_sen2 = self.fid_newTrain.create_dataset('sen2', shape=(1, 32, 32, 10), maxshape=(None, 32, 32, 10),chunks=(1, 32, 32, 10))
        # fid_newTrain_label = self.fid_newTrain.create_dataset('label', shape=(1, 17), maxshape=(None,17),chunks=(1,17))
        sen1_list=[]
        sen2_list = []
        label_list = []
        for i in range(self.fid_original_training['sen1'].shape[0]):
            if list(orig_label[i]).index(1) in indexList:
                sen1_list.append(list(orig_sen1[i]))
                sen2_list.append(list(orig_sen2[i]))
                label_list.append(list(orig_label[i]))
        self.fid_newTrain['sen1']=np.array(sen1_list)
        self.fid_newTrain['sen2'] = np.array(sen2_list)
        self.fid_newTrain['label'] = np.array(label_list)
        self.fid_newTrain.close()

    def get_sen1_data(self, bias, img_list=[], counts=1, train=True, test=False, label=0):
        s1_data = None
        if test:
            if label == 1:
                s1_data = self.fid_test_2['sen1']
            else:
                s1_data = self.fid_test['sen1']
        elif train:
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

    def get_sen2_data(self, bias, img_list=[], counts=1, train=True, test=False, label=0):
        s2_data = None
        if test:
            if label == 1:
                s2_data = self.fid_test_2['sen2']
            else:
                s2_data = self.fid_test['sen2']
            
        elif train:
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

    def get_all_data(self, bias, sen1_list=[], sen2_list=[], counts=1, train=True, test=False):
        s1_data = None
        s2_data = None
        if test:
            s1_data = self.fid_test['sen1']
            s2_data = self.fid_test['sen2']
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
            # print(label)
        else:
            label = self.fid_validation['label']
        # print(label.shape)
        res_list = []
        # print("get_label")
        # print(counts)
    
        for item in range(counts):
            per_item = []
            # print(item)
            # print(int(bias)+int(item))
            this_item = label[int(bias)+int(item)]
        print(this_item)
            
        res_mat = np.array(res_list)
        # print(type(res_mat))
        return res_mat
    
    def get_op_label(self, bias, counts=1, train=True, test=False):
        label = None
        if train:
            label = self.fid_training['label']
        else:
            label = self.fid_validation['label']
        re_list = []
        for item in range(counts):
            this_item = label[int(bias)+int(item)-1]
            # print(this_item)
            flag = 0
            for i in this_item:
                if int(i) == 1:
                    break
                else:
                    flag += 1
            re_list.append(flag)

        return re_list

    def op_mat(self, mat, method="zero-order"):
        # print(type(mat))
        if method == "zero-order":
            # tensor_mat = t.Tensor(mat)
            # no_mat = F.normalize(tensor_mat).numpy()
            miu = np.average(mat)
            sigma = np.std(mat)
            nor_mat = (mat - miu)/sigma
            # print(nor_mat)
            return nor_mat
        else:
            min_v = np.min(mat)
            max_v = np.max(mat)
            nor_mat = (mat - min_v)/(max_v - min_v)
            return nor_mat

    # todo
    def convolution(self,input_matrix,num_conv=8, is_resize=True,resize=8):
        if is_resize:
            mat = cv2.resize(input_matrix, (int(resize),int(resize)),0,0)
            
        else:
            conv_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            mat=input_matrix
            for i in range(num_conv):
                mat = signal.convolve2d(mat, conv_kernel, mode='same')
                size = mat.shape[0]
                mat = mat[1:size - 1, 1:size - 1]
            
        return mat

    def robinson(self, input_matrix):
        img_list = []
        k1 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
        k2 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
        k3 = np.array([[-3,5,5],[-3,0,5],[5,5,5]])
        k4 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
        k5 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
        k6 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
        k7 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
        k8 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
        input_k1 = signal.convolve2d(input_matrix, k1, mode='same', boundary='fill')
        img_list.append(abs(input_k1))
        input_k2 = signal.convolve2d(input_matrix, k2, mode='same', boundary='fill')
        img_list.append(abs(input_k2))
        input_k3 = signal.convolve2d(input_matrix, k3, mode='same', boundary='fill')
        img_list.append(abs(input_k3))
        input_k4 = signal.convolve2d(input_matrix, k4, mode='same', boundary='fill')
        img_list.append(abs(input_k4))
        input_k5 = signal.convolve2d(input_matrix, k5, mode='same', boundary='fill')
        img_list.append(abs(input_k5))
        input_k6 = signal.convolve2d(input_matrix, k6, mode='same', boundary='fill')
        img_list.append(abs(input_k6))
        input_k7 = signal.convolve2d(input_matrix, k7, mode='same', boundary='fill')
        img_list.append(abs(input_k7))
        input_k8 = signal.convolve2d(input_matrix, k8, mode='same', boundary='fill')
        img_list.append(abs(input_k8))
        edge = img_list[0]
        for i in range(len(img_list)):
            edge = edge*(edge>=img_list[i]) +img_list[i]*(edge<img_list[i])
        return edge
    
    def blur(self, input_matrix):
        input_matrix_1 = cv2.GaussianBlur(input_matrix, (3,3),3)
        return input_matrix_1
        



    def get_ND_mat(self, bias, resize=None, counts=1,num_conv=8,conv=True, train=True, test=False, label=0):
        re_mat = None
        if resize is None:
            
            sen1_data_0 = self.get_sen1_data(bias=bias, img_list=[0], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_1 = self.get_sen1_data(bias=bias, img_list=[1], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_2 = self.get_sen1_data(bias=bias, img_list=[2], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_3 = self.get_sen1_data(bias=bias, img_list=[3], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_4 = self.get_sen1_data(bias=bias, img_list=[4], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_5 = self.get_sen1_data(bias=bias, img_list=[5], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_6 = self.get_sen1_data(bias=bias, img_list=[6], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen1_data_7 = self.get_sen1_data(bias=bias, img_list=[7], counts=counts, train=train, test=test, label=label)[0][0,:,:]            
            sen2_data_0 = self.get_sen2_data(bias=bias, img_list=[0], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen2_data_1 = self.get_sen2_data(bias=bias, img_list=[1], counts=counts, train=train, test=test, label=label)[0][0,:,:]  
            sen2_data_2 = self.get_sen2_data(bias=bias, img_list=[2], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            # sen2_data_3 = self.get_sen2_data(bias=bias, img_list=[3], counts=counts, train=train, test=test)[0][0,:,:]
            # sen2_data_4 = self.get_sen2_data(bias=bias, img_list=[4], counts=counts, train=train, test=test)[0][0,:,:]
            # sen2_data_5 = self.get_sen2_data(bias=bias, img_list=[5], counts=counts, train=train, test=test)[0][0,:,:]
            sen2_data_6 = self.get_sen2_data(bias=bias, img_list=[6], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen2_data_7 = self.get_sen2_data(bias=bias, img_list=[7], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen2_data_8 = self.get_sen2_data(bias=bias, img_list=[8], counts=counts, train=train, test=test, label=label)[0][0,:,:]
            sen2_data_9 = self.get_sen2_data(bias=bias, img_list=[9], counts=counts, train=train, test=test, label=label)[0][0,:,:]

            sen1_data_0_f = sen1_data_0.flatten()
            sen1_data_1_f = sen1_data_1.flatten()
            sen1_data_2_f = sen1_data_2.flatten()
            sen1_data_3_f = sen1_data_3.flatten()
            sen1_data_6_f = sen1_data_6.flatten()
            sen1_data_7_f = sen1_data_7.flatten()
            sen1_data_norm_1 = np.sqrt(np.square(sen1_data_0_f) + np.square(sen1_data_1_f))
            sen1_data_norm_2 = np.sqrt(np.square(sen1_data_2_f) + np.square(sen1_data_3_f))
            sen1_data_norm_3 = np.sqrt(np.square(sen1_data_6_f) + np.square(sen1_data_7_f))
            sen1_data_exp_1 = 1 - np.exp(np.negative(sen1_data_norm_1))
            sen1_data_exp_2 = 1 - np.exp(np.negative(sen1_data_norm_2))
            sen1_data_exp_3 = 1 - np.exp(np.negative(sen1_data_norm_3))
            sen1_data_exp_mat_1 = sen1_data_exp_1.reshape(sen1_data_0.shape)
            sen1_data_exp_mat_2 = sen1_data_exp_2.reshape(sen1_data_0.shape)
            sen1_data_exp_mat_3 = sen1_data_exp_3.reshape(sen1_data_0.shape)
            size = sen1_data_0.shape[0]
            reshape_sen2_data_0 = sen2_data_0.reshape((1,size,size))
            reshape_sen2_data_1 = sen2_data_1.reshape((1,size,size))
            reshape_sen2_data_2 = sen2_data_2.reshape((1,size,size))
            reshape_sen2_data_6 = sen2_data_6.reshape((1,size,size))
            reshape_sen2_data_7 = sen2_data_7.reshape((1,size,size))
            reshape_sen2_data_8 = sen2_data_8.reshape((1,size,size))
            reshape_sen2_data_9 = sen2_data_9.reshape((1,size,size))
            reshape_sen1_data_4 = sen1_data_4.reshape((1,size,size))
            reshape_sen1_data_5 = sen1_data_5.reshape((1,size,size))
            reshape_sen1_data_exp_mat_1 = sen1_data_exp_mat_1.reshape((1,size,size))
            reshape_sen1_data_exp_mat_2 = sen1_data_exp_mat_2.reshape((1,size,size))
            reshape_sen1_data_exp_mat_3 = sen1_data_exp_mat_3.reshape((1,size,size))

            nor_reshape_sen2_data_0 = self.op_mat(reshape_sen2_data_0, method="zero-order")
            nor_reshape_sen2_data_1 = self.op_mat(reshape_sen2_data_1, method="zero-order")
            nor_reshape_sen2_data_2 = self.op_mat(reshape_sen2_data_2, method="zero-order")
            nor_reshape_sen2_data_6 = self.op_mat(reshape_sen2_data_6, method="zero-order")
            nor_reshape_sen2_data_7 = self.op_mat(reshape_sen2_data_7, method="zero-order")
            nor_reshape_sen2_data_8 = self.op_mat(reshape_sen2_data_8, method="zero-order")
            nor_reshape_sen2_data_9 = self.op_mat(reshape_sen2_data_9, method="zero-order")
            nor_reshape_sen1_data_4 = self.op_mat(reshape_sen1_data_4, method="zero-order")
            nor_reshape_sen1_data_5 = self.op_mat(reshape_sen1_data_5, method="zero-order")
            nor_reshape_sen1_data_exp_mat_1 = self.op_mat(reshape_sen1_data_exp_mat_1, method="zero-order")
            nor_reshape_sen1_data_exp_mat_2 = self.op_mat(reshape_sen1_data_exp_mat_2, method="zero-order")
            nor_reshape_sen1_data_exp_mat_3 = self.op_mat(reshape_sen1_data_exp_mat_3, method="zero-order")

            if conv:
                # 调用自定义convolution函数，每次调用各维度均减小2，连续调用8次，将32*32减小为16*16
                sen1_nor_reshape_conv_4 = self.blur(nor_reshape_sen1_data_4[0])
                sen1_nor_reshape_conv_4 = self.convolution(sen1_nor_reshape_conv_4)
                sen1_nor_reshape_conv_5 = self.blur(nor_reshape_sen1_data_5[0]) 
                sen1_nor_reshape_conv_5 = self.convolution(sen1_nor_reshape_conv_5)
                sen1_nor_reshape_exp_conv_1 = self.blur(nor_reshape_sen1_data_exp_mat_1[0]) 
                sen1_nor_reshape_exp_conv_1 = self.convolution(sen1_nor_reshape_exp_conv_1)
                sen1_nor_reshape_exp_conv_2 = self.blur(nor_reshape_sen1_data_exp_mat_2[0]) 
                sen1_nor_reshape_exp_conv_2 = self.convolution(sen1_nor_reshape_exp_conv_2)
                sen1_nor_reshape_exp_conv_3 = self.blur(nor_reshape_sen1_data_exp_mat_3[0]) 
                sen1_nor_reshape_exp_conv_3 = self.convolution(sen1_nor_reshape_exp_conv_3)

                sen2_nor_reshape_conv_0 = self.robinson(nor_reshape_sen2_data_0[0])
                sen2_nor_reshape_conv_1 = self.robinson(nor_reshape_sen2_data_1[0])
                sen2_nor_reshape_conv_2 = self.robinson(nor_reshape_sen2_data_2[0])
                sen2_nor_reshape_conv_6 = self.robinson(nor_reshape_sen2_data_6[0])
                sen2_nor_reshape_conv_7 = self.robinson(nor_reshape_sen2_data_7[0])
                sen2_nor_reshape_conv_8 = self.robinson(nor_reshape_sen2_data_8[0])
                sen2_nor_reshape_conv_9 = self.robinson(nor_reshape_sen2_data_9[0])
                
                sen2_nor_reshape_conv_0 = self.convolution(sen2_nor_reshape_conv_0)
                sen2_nor_reshape_conv_1 = self.convolution(sen2_nor_reshape_conv_1)
                sen2_nor_reshape_conv_2 = self.convolution(sen2_nor_reshape_conv_2)
                sen2_nor_reshape_conv_6 = self.convolution(sen2_nor_reshape_conv_6)
                sen2_nor_reshape_conv_7 = self.convolution(sen2_nor_reshape_conv_7)
                sen2_nor_reshape_conv_8 = self.convolution(sen2_nor_reshape_conv_8)
                sen2_nor_reshape_conv_9 = self.convolution(sen2_nor_reshape_conv_9)


                newsize = sen2_nor_reshape_conv_0.shape[0]
                nor_reshape_sen2_data_0 = sen2_nor_reshape_conv_0.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_1 = sen2_nor_reshape_conv_1.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_2 = sen2_nor_reshape_conv_2.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_6 = sen2_nor_reshape_conv_6.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_7 = sen2_nor_reshape_conv_7.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_8 = sen2_nor_reshape_conv_8.reshape((1, newsize, newsize))
                nor_reshape_sen2_data_9 = sen2_nor_reshape_conv_9.reshape((1, newsize, newsize))
                nor_reshape_sen1_data_4 = sen1_nor_reshape_conv_4.reshape((1, newsize, newsize))
                nor_reshape_sen1_data_5 = sen1_nor_reshape_conv_5.reshape((1, newsize, newsize))
                nor_reshape_sen1_data_exp_mat_1 = sen1_nor_reshape_exp_conv_1.reshape((1, newsize, newsize))
                nor_reshape_sen1_data_exp_mat_2 = sen1_nor_reshape_exp_conv_2.reshape((1, newsize, newsize))
                nor_reshape_sen1_data_exp_mat_3 = sen1_nor_reshape_exp_conv_3.reshape((1, newsize, newsize))


            re_mat = np.concatenate((nor_reshape_sen2_data_0,nor_reshape_sen2_data_1,nor_reshape_sen2_data_2,\
            nor_reshape_sen2_data_6,nor_reshape_sen2_data_7,nor_reshape_sen2_data_8,nor_reshape_sen2_data_9,\
            nor_reshape_sen1_data_4,nor_reshape_sen1_data_5,nor_reshape_sen1_data_exp_mat_1,\
            nor_reshape_sen1_data_exp_mat_2,nor_reshape_sen1_data_exp_mat_3), axis=0)


            self.op_mat(sen1_data_0)
            del sen2_data_0, sen2_data_1, sen2_data_2, sen2_data_6, \
            sen2_data_7, sen2_data_8, sen2_data_9, sen1_data_0, sen1_data_1, \
            sen1_data_2, sen1_data_3, sen1_data_4, sen1_data_5, sen1_data_6, \
            sen1_data_7, sen1_data_exp_mat_1, sen1_data_exp_mat_2, sen1_data_exp_mat_3,\
            sen1_data_norm_1, sen1_data_norm_2, sen1_data_norm_3, \
            sen1_data_exp_1, sen1_data_exp_2, sen1_data_exp_3, sen1_data_0_f, sen1_data_1_f,\
            sen1_data_2_f, sen1_data_3_f, sen1_data_6_f, sen1_data_7_f
            
            del reshape_sen2_data_0,reshape_sen2_data_1,reshape_sen2_data_2,\
            reshape_sen2_data_6,reshape_sen2_data_7,reshape_sen2_data_8,\
            reshape_sen2_data_9,reshape_sen1_data_4,reshape_sen1_data_5,\
            reshape_sen1_data_exp_mat_1,reshape_sen1_data_exp_mat_2,reshape_sen1_data_exp_mat_3

            del nor_reshape_sen2_data_0,nor_reshape_sen2_data_1,nor_reshape_sen2_data_2,\
            nor_reshape_sen2_data_6,nor_reshape_sen2_data_7,nor_reshape_sen2_data_8,\
            nor_reshape_sen2_data_9,nor_reshape_sen1_data_4,nor_reshape_sen1_data_5,\
            nor_reshape_sen1_data_exp_mat_1,nor_reshape_sen1_data_exp_mat_2,nor_reshape_sen1_data_exp_mat_3

        else:
            pass
        # print(re_mat.shape)
        return re_mat

        # if test:
        #     print("test")
        #     all_sen2_data = self.get_sen2_data(bias=bias,img_list=[],counts=counts, train=train, test=test)

        # elif train:
        #     print("train")
        #     all_sen2_data = self.get_sen2_data(bias=bias,img_list=[],counts=counts, train=train, test=test)
        # else:
        #     print("val")
        #     all_sen2_data = self.get_sen2_data(bias=bias,img_list=[],counts=counts, train=train, test=test)
    def new_get_3D_mat(self, bias, resize, counts=1, train=True, test=False):

        if test:
            b = self.get_sen2_data(bias,[0],1,True)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,True)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,True)[0][0,:,:]
            nir = self.get_sen2_data(bias,[6],1,True)[0][0,:,:]

            


            resize_b = cv2.resize(b, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_g = cv2.resize(g, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_r = cv2.resize(r, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_nir = cv2.resize(nir, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)

            del b,g,r,nir

            reshape_b = resize_b.reshape(1,int(resize),int(resize))
            reshape_g = resize_b.reshape(1,int(resize),int(resize))
            reshape_r = resize_b.reshape(1,int(resize),int(resize))
            reshape_nir = resize_b.reshape(1,int(resize),int(resize))

            del resize_b, resize_g, resize_r, resize_nir

            re_mat = np.concatenate((reshape_b, reshape_g, reshape_b, reshape_nir), axis=0)

            del reshape_b, reshape_g, reshape_r, reshape_nir

            return re_mat

            

        elif train:
            b = self.get_sen2_data(bias,[0],1,True)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,True)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,True)[0][0,:,:]
            nir = self.get_sen2_data(bias,[6],1,True)[0][0,:,:]

            resize_b = cv2.resize(b, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_g = cv2.resize(g, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_r = cv2.resize(r, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_nir = cv2.resize(nir, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)

            del b,g,r,nir

            reshape_b = resize_b.reshape(1,int(resize),int(resize))
            reshape_g = resize_b.reshape(1,int(resize),int(resize))
            reshape_r = resize_b.reshape(1,int(resize),int(resize))
            reshape_nir = resize_b.reshape(1,int(resize),int(resize))

            del resize_b, resize_g, resize_r, resize_nir

            re_mat = np.concatenate((reshape_b, reshape_g, reshape_b, reshape_nir), axis=0)

            del reshape_b, reshape_g, reshape_r, reshape_nir

            return re_mat
            # print(np.concatenate((resize_b,resize_g, resize_nir), axis=1).shape)
        else:
            b = self.get_sen2_data(bias,[0],1,False)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,False)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,False)[0][0,:,:]
            nir = self.get_sen2_data(bias,[6],1,False)[0][0,:,:]

            resize_b = cv2.resize(b, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_g = cv2.resize(g, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_r = cv2.resize(r, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)
            resize_nir = cv2.resize(nir, (int(resize),int(resize)),0,0, interpolation = cv2.INTER_CUBIC)

            del b,g,r,nir

            reshape_b = resize_b.reshape(1,int(resize),int(resize))
            reshape_g = resize_b.reshape(1,int(resize),int(resize))
            reshape_r = resize_b.reshape(1,int(resize),int(resize))
            reshape_nir = resize_b.reshape(1,int(resize),int(resize))

            del resize_b, resize_g, resize_r, resize_nir

            re_mat = np.concatenate((reshape_b, reshape_g, reshape_b, reshape_nir), axis=0)

            del reshape_b, reshape_g, reshape_r, reshape_nir

            return re_mat

    def new_get_length(self, train=True, test=False):
        pass
    def get_length(self, train=True, test=False,label=0):
        if test:
            if label == 1:
                return self.fid_test_2['sen1'].shape[0]
            else:
                # print(f"length:{self.fid_test['sen1'].shape[0]}")
                print("length:{}".format(self.fid_test['sen1'].shape[0]))
                return self.fid_test['sen1'].shape[0]            
        elif train:
            return self.fid_training['label'].shape[0]
        else:
            return self.fid_validation['label'].shape[0]


    def get_transformmat_to_img(self, bias, counts=1, train=True, test=False):
        if test:
            b = self.get_sen2_data(bias,[0],1,True)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,True)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,True)[0][0,:,:]
            # nir = self.get_sen2_data(bias,[6],1,True)[0][0,:,:]
            # new_img_rgb_nir = ((0.25*b+0.33*g+0.2*r)+nir)/4
            gray = 0.2989 * r * 255 + 0.5870 * g * 255 + 0.1140 * b * 255

            re_img = Image.fromarray(gray.astype('uint8')).convert('RGB')
           
            return re_img

        elif train:

            b = self.get_sen2_data(bias,[0],1,True)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,True)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,True)[0][0,:,:]
            # nir = self.get_sen2_data(bias,[6],1,True)[0][0,:,:]
            # new_img_rgb_nir = ((0.25*b+0.33*g+0.2*r)+nir)/4
            gray = 0.2989 * r*255 + 0.5870 * g*255 + 0.1140 * b*255


            re_img = Image.fromarray(gray.astype('uint8')).convert('RGB')
           
            return re_img

        else:
            b = self.get_sen2_data(bias,[0],1,False)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,False)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,False)[0][0,:,:]
            # nir = self.get_sen2_data(bias,[6],1,False)[0][0,:,:]
            # new_img_rgb_nir = ((0.25*b+0.33*g+0.2*r)+nir)/4
            gray = 0.2989 * r*255 + 0.5870 * g*255 + 0.1140 * b*255
            re_img = Image.fromarray(gray.astype('uint8')).convert('RGB')
            return re_img


class readData1(object):
    def __init__(self):
        self.base_dir = "./data"
        # self.base_dir = "../data"
        # print(os.listdir(self.base_dir))
        # self.base_dir = "../../../data"
        # self.path_training = os.path.join(self.base_dir, "training.h5")
        self.path_validation = os.path.join(self.base_dir,"validation.h5")
        self.path_training = os.path.join(self.base_dir,"mini_data/training/training.00.h5")
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
            label = self.fid_validation['label']
        # print(label.shape)
        res_list = []
        print("get_label")
        print(counts)
    
        for item in range(counts):
            per_item = []
            print(item)
            print(int(bias)+int(item))
            this_item = label[int(bias)+int(item)]
            
        res_mat = np.array(res_list)
        # print(type(res_mat))
        return res_mat
    
    def get_op_label(self, bias, counts=1, train=True):
        label = None
        if train:
            label = self.fid_training['label']
            # print("train")
        else:
            label = self.fid_validation['label']
            # print("not train")
        # print(train)
        # print(label)
        # print("get_op_label")
        # print(counts)
        # print(label.shape)
        re_list = []
        for item in range(counts):
            # print(item)
            # print(int(bias)+int(item))
            this_item = label[int(bias)+int(item)-1]
            # print(this_item)
            flag = 0
            for i in this_item:
                if int(i) == 1:
                    break
                else:
                    flag += 1
            re_list.append(flag)
        # print(re_list)
        return re_list



    def get_length(self, train=True):
        if train:
            return self.fid_training['label'].shape[0]
        else:
            return self.fid_validation['label'].shape[0]


    def get_transform_to_img(self, bias, counts=1, train=True):
        if train:

            b = self.get_sen2_data(bias,[0],1,True)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,True)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,True)[0][0,:,:]
            # nir = self.get_sen2_data(bias,[6],1,True)[0][0,:,:]
            # new_img_rgb_nir = ((0.25*b+0.33*g+0.2*r)+nir)/4
            gray = 0.2989 * r*255 + 0.5870 * g*255 + 0.1140 * b*255


            re_img = Image.fromarray(gray.astype('uint8')).convert('RGB')
           
            return re_img

        else:
            b = self.get_sen2_data(bias,[0],1,False)[0][0,:,:]
            g = self.get_sen2_data(bias,[1],1,False)[0][0,:,:]
            r = self.get_sen2_data(bias,[2],1,False)[0][0,:,:]
            # nir = self.get_sen2_data(bias,[6],1,False)[0][0,:,:]
            # new_img_rgb_nir = ((0.25*b+0.33*g+0.2*r)+nir)/4
            gray = 0.2989 * r*255 + 0.5870 * g*255 + 0.1140 * b*255
            re_img = Image.fromarray(gray.astype('uint8')).convert('RGB')

            return re_img

#data = a.get_ND_mat(0,None)
# length = a.get_length(True)
# label = a.get_op_label(0, 1, False)
# print(label[0])

if __name__ == '__main__':
    # zhongshiang
    a = readData()
    a.get_ND_mat(0)
    # a.generateDataset([0,6,11,14])
    # fid_newTrain = h5py.File(a.path_newTrain, 'r')
    # newTrain_sen1=fid_newTrain['sen1']
    # newTrain_sen2 = fid_newTrain['sen2']
    # newTrain_label = fid_newTrain['label']
    # print(fid_newTrain.keys())
    # print(newTrain_sen1.shape)
    # print(newTrain_sen2.shape)
    # print(newTrain_label.shape)


    # sen1_data_4 = a.get_sen1_data(0, [4], 1, True, False)[0][0,:,:]
    # size = sen1_data_4.shape[0]
    # reshape_sen1_data_4 = sen1_data_4.reshape((1,size,size))
    # nor_reshape_sen1_data_4 = a.op_mat(reshape_sen1_data_4, method="zero-order")
    # sen1_nor_reshape_conv_4=a.convolution(nor_reshape_sen1_data_4[0],8)
    # print(sen1_nor_reshape_conv_4.shape)
