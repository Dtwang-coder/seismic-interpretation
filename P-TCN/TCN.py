#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

 
#残差块

def ResBlock(x,filters,kernel_size,dilation_rate):

    r=tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,kernel_initializer='he_normal'))(x) #第一卷积
    
    r=Activation('relu')(r)
    
    r=Dropout(0.2)(r)

    r=tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,kernel_initializer='he_normal'))(r) #第二卷积
    
    r=Activation('relu')(r)
    
    r=Dropout(0.2)(r)

    if x.shape[-1]==filters:

        shortcut=x

    else:

        shortcut=Conv1D(filters,kernel_size = 1,kernel_initializer='he_normal',padding='same')(x)  #shortcut（捷径）

    o=add([r,shortcut])

    o=Activation('relu')(o) 

    return o


def TCN(pretrained_weights = None,input_size = (462,1)):

    inputs = Input(input_size)

    x=ResBlock(inputs,filters=64,kernel_size=9,dilation_rate=1)
    x=ResBlock(x,filters=64,kernel_size=9,dilation_rate=2)
    x=ResBlock(x,filters=64,kernel_size=9,dilation_rate=4)
    x=ResBlock(x,filters=64,kernel_size=9,dilation_rate=8)
    x=ResBlock(x,filters=64,kernel_size=9,dilation_rate=16)
    x=ResBlock(x,filters=64,kernel_size=9,dilation_rate=32)
#     x=ResBlock(x,filters=128,kernel_size=5,dilation_rate=64)
#     x=ResBlock(x,filters=128,kernel_size=5,dilation_rate=128)
#     x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=256)
#     x=ResBlock(x,filters=128,kernel_size=3,dilation_rate=256)
#     x = concatenate([inputs,x], axis = 2)
    x = Conv1D(10,1,padding='same',activation = 'softmax')(x)
    
#     x = TimeDistributed(Dense(10,activation = 'softmax'))(x)


    model=Model(inputs=inputs,outputs=x)
    
    
    
    def wight_loss(y_true,y_pred):

        weights=np.array([2.35401583,2.16751167,0.57258946,1.69162928,1.13056343,0.69857325,0.65807407,1.1236691,0.63163292,0.90085359])
#         np.array([1,1,1,2.5,2.5,1,1,2.5,1,1.5])
    #         np.array([2.34895431,2.18731289,0.59859472,1.69550651,1.12728132,0.69789008,0.66551538,1.13234289,0.63398755,0.89854526])
    #         [2.27764583,2.18386,0.59770763,1.6972665,1.13155215,0.69313136,0.66566828,1.12859699,0.63659551,0.8977111]
#             np.array([2.35401583,2.16751167,0.57258946,1.69162928,1.13056343,0.69857325,0.65807407,1.1236691,0.63163292,0.90085359])

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) 
        loss = y_true * K.log(y_pred)* weights
        loss = K.mean(-K.sum(loss, -1))
        return loss
    #编译模型
#     model.compile(optimizer = Adam(lr = 0.001),loss = wight_loss, metrics = ['categorical_accuracy'])
    model.compile(optimizer = Adam(lr = 0.001),loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    #查看网络结构

    model.summary()
    
    if(pretrained_weights):

        model.load_weights(pretrained_weights)

    return model

