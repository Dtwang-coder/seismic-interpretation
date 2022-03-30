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
# import tensorflow.keras as K
# import tensorflow as tf


 
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
#     x=ResBlock(x,filters=64,kernel_size=5,dilation_rate=64)
#     x=ResBlock(x,filters=128,kernel_size=5,dilation_rate=128)
#     x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=256)
#     x=ResBlock(x,filters=128,kernel_size=3,dilation_rate=256)
#     x = concatenate([inputs,x], axis = 2)
    x = Conv1D(10,1,padding='same',activation = 'softmax')(x)
    
#     x = TimeDistributed(Dense(10,activation = 'softmax'))(x)


    model=Model(inputs=inputs,outputs=x)

    #编译模型
    

#     def dice_coef(y_true,y_pred,smooth=1):
#         #求得每个sample的每个类的dice
#         intersection = K.sum(y_true * y_pred, axis=(1,2))
#         union = K.sum(y_true, axis=(1,2)) + K.sum(y_pred, axis=(1,2))
#         sample_dices=(2. * intersection + smooth) / (union + smooth) #一维数组 为各个类别的dice
#         #求得每个类的dice
#         dices=K.mean(sample_dices,axis=0)
#         return K.mean(dices) #所有类别dice求平均的dice

#     def dice_coef_loss_fun(y_true,y_pred,smooth=1):
#         return 1-dice_coef(y_true,y_pred,smooth=1)
    


    def generalized_dice(y_true, y_pred,smooth=0):

        # Compute weights: "the contribution of each label is corrected by the inverse of its volume"

        w = K.sum(y_true, axis=(0, 1,2))

        w = 1 / (w ** 2 + 0.00001)

        # w为各个类别的权重，占比越大，权重越小

        # Compute gen dice coef:

        numerator = y_true * y_pred

        numerator = w * K.sum(numerator, axis=(0, 1,2))

        numerator = K.sum(numerator)



        denominator = y_true + y_pred

        denominator = w * K.sum(denominator, axis=(0, 1,2))

        denominator = K.sum(denominator)



        gen_dice_coef = numerator / denominator



        return  2 * gen_dice_coef


    def generalized_dice_loss_fun(y_true, y_pred,smooth=0):
        
        return 1 - generalized_dice(y_true, y_pred,smooth=0)
    


    def wight_loss(y_true,y_pred):
        
        weights=np.array([1,1,1,2.5,2.5,1,1,2.5,1,1.5])
#         np.array([2.34895431,2.18731289,0.59859472,1.69550651,1.12728132,0.69789008,0.66551538,1.13234289,0.63398755,0.89854526])
#         [2.27764583,2.18386,0.59770763,1.6972665,1.13155215,0.69313136,0.66566828,1.12859699,0.63659551,0.8977111]
#         np.array([2.35401583,2.16751167,0.57258946,1.69162928,1.13056343,0.69857325,0.65807407,1.1236691,0.63163292,0.90085359])
#         weights = K.variable(weights)
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) 
        loss = y_true * K.log(y_pred)* weights
        loss = K.mean(-K.sum(loss, -1))
        return loss
    
    


    def dice(y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=(1))
        union = K.sum(y_true, axis=(1))+K.sum(y_pred, axis=(1))
        sample_dices=(2. * intersection +0.001) / (union +0.001)
        dices=K.mean(sample_dices,axis=0)
        return K.mean(dices)

    def dice_loss(y_true, y_pred):
        return 1-dice(y_true, y_pred)
    
# y_true and y_pred should be one-hot

# y_true.shape = (None,Width,Height,Channel)

# y_pred.shape = (None,Width,Height,Channel)

    def dice_coef(y_true, y_pred, smooth=0.01):

        mean_loss = 0;

        for i in range(y_pred.shape[-1]):

            intersection = K.sum(y_true[:,:,i] * y_pred[:,:,i], axis=(1))

            union = K.sum(y_true[:,:,i], axis=(1)) + K.sum(y_pred[:,:,i], axis=(1))

        mean_loss += (2. * intersection + smooth) / (union + smooth)

        return K.mean(mean_loss, axis=0)

    def dice_coef_loss(y_true, y_pred):

        return 1 - dice_coef(y_true, y_pred, smooth=0.01)

    
    def gen_dice_loss(y_true,y_pred):
        y_true_f = K.reshape(y_true,shape=(-1,4))
        y_pred_f = K.reshape(y_pred,shape=(-1,4))
        sum_p=K.sum(y_pred_f,axis=-2)
        sum_r=K.sum(y_true_f,axis=-2)
        sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
        weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
        generalised_dice_numerator =2*K.sum(weights*sum_pr)
        generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
        generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
        GDL=1-generalised_dice_score
        del sum_p,sum_r,sum_pr,weights
        return GDL



#     def dice_coef(y_true,y_pred):
#         sum1 = 2*tf.reduce_sum(y_true*y_pred,axis=(1,2))
#         sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(1,2))
#         dice = (sum1+0.1)/(sum2+0.1)
#         dice = tf.reduce_mean(dice)
#         return dice
#     def dice_coef_loss(y_true,y_pred):
#         return 1.-dice_coef(y_true,y_pred)

    model.compile(optimizer = Adam(lr = 0.001),loss = ['categorical_crossentropy'], metrics = ['categorical_accuracy'])
#     model.compile(optimizer = Adam(lr = 0.001),loss = wight_loss, metrics = ['categorical_accuracy'])
    
    #查看网络结构

    model.summary()
    
    if(pretrained_weights):

        model.load_weights(pretrained_weights)

    return model

