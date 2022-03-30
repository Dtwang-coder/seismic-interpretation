#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import os
from skimage import io
import cv2 as cv
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.utils import shuffle
from os.path import join as pjoin


def seismic_data(input_dir,traces):
    a = os.listdir(input_dir)
    count=0
    seismic = []
    seismic1 = np.zeros(shape=(len(a),io.imread(os.path.join(input_dir,a[0])).shape[0],
              traces))
    seismic2 = np.zeros(shape=(len(a),io.imread(os.path.join(input_dir,a[16])).shape[0],
              traces))
    for i in a:
        seismic_data=io.imread(os.path.join(input_dir,i),-1)
#         print(seismic_data)
#         seismic_data = DataFrame(seismic_data)
        data_index1 = np.linspace(0, 650, traces).astype(int)
        data_index2 = np.linspace(0, 950, traces).astype(int)
        if seismic_data.shape[1] == 651:
            seismic1[count] = np.array(seismic_data[:,data_index1])
        else:
            seismic2[count] = np.array(seismic_data[:,data_index2])
        count+=1
    seismic =  np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],
                         seismic1[9],seismic2[10],seismic2[11],seismic2[12],seismic2[13],seismic2[14],seismic2[15],seismic2[16]))
#     np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],seismic1[9],seismic1[10],seismic1[11],seismic1[12],seismic1[13],seismic1[14],seismic1[15],seismic1[16],seismic1[17],seismic1[18],seismic2[19],seismic2[20],seismic2[21],seismic2[22],seismic2[23],seismic2[24],seismic2[25],seismic2[26],seismic2[27],seismic2[28],seismic2[29],seismic2[30],seismic2[31]))
# np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],
#                          seismic1[9],seismic2[10],seismic2[11],seismic2[12],seismic2[13],seismic2[14],seismic2[15],seismic2[16]))
# np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],seismic1[9],seismic1[10],seismic1[11],seismic1[12],seismic1[13],seismic1[14],seismic1[15],seismic1[16],seismic1[17],seismic1[18],seismic2[19],seismic2[20],seismic2[21],seismic2[22],seismic2[23],seismic2[24],seismic2[25],seismic2[26],seismic2[27],seismic2[28],seismic2[29],seismic2[30],seismic2[31]))
#     np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],
#                          seismic1[9],seismic2[10],seismic2[11],seismic2[12],seismic2[13],seismic2[14],seismic2[15],seismic2[16]))
# np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],seismic1[9],seismic1[10],seismic1[11],seismic1[12],seismic1[13],seismic1[14],seismic1[15],seismic1[16],seismic1[17],seismic1[18],seismic2[19],seismic2[20],seismic2[21],seismic2[22],seismic2[23],seismic2[24],seismic2[25],seismic2[26],seismic2[27],seismic2[28],seismic2[29],seismic2[30],seismic2[31]))
# np.hstack((seismic1[0],seismic1[1],seismic1[2],seismic1[3],seismic1[4],seismic1[5],seismic1[6],seismic1[7],seismic1[8],
#                          seismic1[9],seismic2[10],seismic2[11],seismic2[12],seismic2[13],seismic2[14],seismic2[15],seismic2[16]))
    print(seismic.shape)
#     print(seismic)
    return seismic


def train_val_split(seismic,seismic_mask):
        
#     np.random.seed(2018)
#     valid_index = np.linspace(0, 13166, 2007).astype(int)#设为验证集index
#     train_index = np.setdiff1d(np.arange(0, 13166).astype(int), valid_index)#训练集index
    
#     valid_index = shuffle(valid_index)
#     train_index = shuffle(train_index)
    
    x_tra, y_tra = np.array(seismic.T), np.array(seismic_mask.T)
#     x_val, y_val = np.array(seismic.T), np.array(seismic_mask.T)
    
    x_train_re = np.reshape(x_tra, (x_tra.shape[0],x_tra.shape[1],1))
    y_train_re = np.reshape(y_tra, (y_tra.shape[0],y_tra.shape[1],1))

#     x_valid_re = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))
#     y_valid_re = np.reshape(y_val, (y_val.shape[0],y_val.shape[1],1))

    y_tra = adjustData(y_train_re)
#     x_val,y_val = adjustData(x_valid_re,y_valid_re)
    
    print(x_tra.shape,y_tra.shape)
    
    return x_tra, y_tra



def adjustData(mask):

#     img = (img - img.mean()) / img.std()
#     img = (img-img.min())/(img.max()-img.min())

    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]

    new_mask = np.zeros(mask.shape + (10,))

    new_mask[mask == 0.,   0] = 1
    new_mask[mask == 28.,   1] = 1
    new_mask[mask == 57.,   2] = 1
    new_mask[mask == 85.,   3] = 1
    new_mask[mask == 113.,   4] = 1
    new_mask[mask == 142.,   5] = 1
    new_mask[mask == 170.,   6] = 1
    new_mask[mask == 198.,   7] = 1
    new_mask[mask == 227.,   8] = 1
    new_mask[mask == 255.,   9] = 1
    
    mask = new_mask
    
    return mask

##彩色背景成图##
_background_ = [128,138,135]
layer1 = [156,102,31]
layer2 = [127,255,212]
layer3 = [240,230,140]
layer4 = [60,90,170]
layer5 = [176,224,230]
layer6 = [244,164,96]
layer7 = [255,192,203]
layer8 = [135,206,235]
salt = [255,128,10]

color_dict1 = np.array([_background_, layer1, layer2, layer3, layer4,layer5,layer6,layer7,layer8,salt])

##灰度像素值
gray1 = [0,0,0]
gray2 = [28,28,28]
gray3 = [57,57,57]
gray4 = [85,85,85]
gray5 = [113,113,113]
gray6 = [142,142,142]
gray7 = [170,170,170]
gray8 = [198,198,198]
gray9 = [227,227,227]
gray10 = [255,255,255]

color_dict2 = np.array([gray1,gray2,gray3,gray4,gray5,gray6,gray7,gray8,gray9,gray10])


def testGenerator(test_path):
    
    a=os.listdir(test_path)
    for i in a:
        seismic = cv.imread(pjoin(test_path, i),-1)
        seismic = DataFrame(seismic)
        
        test_data = np.array(seismic.T)
        test_data = np.reshape(test_data, (test_data.shape[0],test_data.shape[1],1))
#         test_data = (test_data - test_data.mean()) / test_data.std()
        print(test_data.shape)
        yield test_data
        
##彩色成图        
def labelVisualize_rgb(color_dict1,img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict1[index_of_class]
    return img_out
##灰度图
def labelVisualize_gray(color_dict2,img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict2[index_of_class]
    return img_out

def saveResult_in_rgb(save_path,npyfile,args,test_path):
    a=os.listdir(test_path)
    for i in range(0,args):
        img = npyfile[951*i:951*(i+1),:,:]
        img_out = labelVisualize_rgb(color_dict1,img)
        img_out = img_out.astype(np.uint8)
        image=cv.transpose(img_out)
        io.imsave(os.path.join(save_path,a[i]),image)
        
def saveResult_cr_rgb(save_path,npyfile,args,test_path):
    a=os.listdir(test_path)
    for i in range(0,args):
        img = npyfile[651*i:651*(i+1),:,:]
        img_out = labelVisualize_rgb(color_dict1,img)
        img_out = img_out.astype(np.uint8)
        image=cv.transpose(img_out)
        io.imsave(os.path.join(save_path,a[i]),image)
        
        
def saveResult_in_gray(save_path,npyfile,args,test_path):
    a=os.listdir(test_path)
    for i in range(0,args):
        img = npyfile[951*i:951*(i+1),:,:]
        img_out = labelVisualize_gray(color_dict2,img)
        img_out = img_out.astype(np.uint8)
        image=cv.transpose(img_out)
        io.imsave(os.path.join(save_path,a[i]),image)
        
def saveResult_cr_gray(save_path,npyfile,args,test_path):
    a=os.listdir(test_path)
    for i in range(0,args):
        img = npyfile[651*i:651*(i+1),:,:]
        img_out = labelVisualize_gray(color_dict2,img)
        img_out = img_out.astype(np.uint8)
        image=cv.transpose(img_out)
        io.imsave(os.path.join(save_path,a[i]),image)




