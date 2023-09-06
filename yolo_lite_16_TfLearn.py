#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:10:30 2020

@author: sbasak
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib
import tflearn

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,recall_score,accuracy_score


import h5py


FFT_frames=256
FFT_N= 256



classes=["dx4e","dx6i","MTx","Nineeg","Parrot","q205","S500","tello","WiFi","wltoys"]
mods = len(classes)

Ylastdim=2*5+mods
maxsig=70
B = 2
C = len(classes)


#dposl=dposl.reshape(dposl.shape[0],maxsig,dposl.shape[1])

shape = FFT_N
grid_nums=16
out_grid_shape=grid_nums
grid = shape/grid_nums
grid_val=grid

nfft=FFT_N


path = 'GenDataYOLO_10_snrtest.h5'
h5f = h5py.File(path, 'r')

X_train_t = h5f['X_train']
Y_train_t = h5f['Y_train']
train_idx_t = h5f['train_idx']
X_test_t = h5f['X_test']
Y_test_t = h5f['Y_test']
test_idx_t = h5f['test_idx']
snrs_train_t= h5f['SNR_train']
snrs_test_t= h5f['SNR_test']
num_sig_presence_test_t=h5f['num_sig_presence_test']

X_train=np.array(X_train_t[()])
X_test=np.array(X_test_t[()])
Y_train=np.array(Y_train_t[()])
Y_test=np.array(Y_test_t[()])
train_idx=np.array(train_idx_t[()])
test_idx=np.array(test_idx_t[()])
snrs_train=np.array(snrs_train_t[()])
snrs_test=np.array(snrs_test_t[()])
num_sig_presence_test=np.array(num_sig_presence_test_t[()])
h5f.close()

print("--"*10)
print("Training data IQ:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data IQ",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*10)


def tf_iou(pxy0,pwh0,txy0,twh0):
    #iou calculation
    #8x8 constant
    mulval = tf.constant([ 0,  32,  64,  96, 128, 160, 192, 224], dtype="float32")

    px0,py0 = tf.split(pxy0,[1,1],axis=3) 
    pw0,ph0 = tf.split(pwh0,[1,1],axis=3)
    px0 = px0 * grid_val + mulval 
    py0 = py0 * grid_val + mulval 
    pw0 = pw0 * 256
    ph0 = ph0 * 256

    tx0,ty0 = tf.split(txy0,[1,1],axis=3) 
    tw0,th0 = tf.split(twh0,[1,1],axis=3)
    tx0 = tx0 * grid_val + mulval
    ty0 = ty0 * grid_val + mulval 
    tw0 = tw0 * 256
    th0 = th0 * 256

    x11,y11,x12,y12 = px0-(pw0/2.0), py0-(ph0/2.0),px0+(pw0/2.0), py0+(ph0/2.0)
    x21,y21,x22,y22 = tx0-(tw0/2.0), ty0-(th0/2.0),tx0+(tw0/2.0), ty0+(th0/2.0)

    xm1 = tf.maximum(x11,x21)
    ym1 = tf.maximum(y11,y21)
    xm2 = tf.minimum(x12,x22)
    ym2 = tf.minimum(y12,y22)
    
    #intersection area
    ia = (xm2-xm1) * (ym2-ym1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - ia
    
    # compute the IoU
    iou = ia/ union_area
    
    return iou

def tf_get_iou(px0,py0,pw0,ph0,tx0,ty0,tw0,th0):
    #iou calculation

    #px0,py0 = tf.split(pxy0,[1,1],axis=3) 
    #pw0,ph0 = tf.split(pwh0,[1,1],axis=3)
    px0 = px0 * grid_val 
    py0 = py0 * grid_val  
    pw0 = pw0 * 256
    ph0 = ph0 * 256

    #tx0,ty0 = tf.split(txy0,[1,1],axis=3) 
    #tw0,th0 = tf.split(twh0,[1,1],axis=3)
    tx0 = tx0 * grid_val 
    ty0 = ty0 * grid_val 
    tw0 = tw0 * 256
    th0 = th0 * 256

    x11,y11,x12,y12 = px0-(pw0/2.0), py0-(ph0/2.0),px0+(pw0/2.0), py0+(ph0/2.0)
    x21,y21,x22,y22 = tx0-(tw0/2.0), ty0-(th0/2.0),tx0+(tw0/2.0), ty0+(th0/2.0)

    xm1 = max(x11,x21)
    ym1 = max(y11,y21)
    xm2 = min(x12,x22)
    ym2 = min(y12,y22)
    
    #intersection area
    ia = (xm2-xm1) * (ym2-ym1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - ia
    
    # compute the IoU
    iou = ia/ union_area
    
    return iou

def my_objective(y_pred, y_true):
    lcord = 5 
    lnoobj =0.5
    with tf.name_scope(None):
        pC0,pxy0,pwh0,pC1,pxy1,pwh1,pc = tf.split(y_pred, [1,2,2,1,2,2,mods], axis = 3)
        tC0,txy0,twh0,tC1,txy1,twh1,tc = tf.split(y_true, [1,2,2,1,2,2,mods], axis = 3)
        
        #iou0 = tf_iou(pxy0,pwh0,txy0,twh0)
        #iou1 = tf_iou(pxy1,pwh1,txy1,twh1)

        loss_xy = lcord * tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(txy0,pxy0)+ tC1 * tf.squared_difference(txy1,pxy1), axis=[1,2,3]))
        #loss_wh = lcord * tf.reduce_mean(tC0 * tf.squared_difference(twh0,pwh0)+ tC1 * tf.squared_difference(twh1,pwh1))
        loss_wh = lcord * tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(tf.sqrt(twh0),tf.sqrt(pwh0))+ tC1 * tf.squared_difference(tf.sqrt(twh1),tf.sqrt(pwh1)), axis=[1,2,3]))
        #loss_Cobj = tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(tC0*iou0,pC0)+ tC1 * tf.squared_difference(tC1*iou1,pC1), axis=[1,2,3]))
        loss_Cobj = tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(tC0,pC0)+ tC1 * tf.squared_difference(tC1,pC1), axis=[1,2,3]))
        loss_Cnoobj = lnoobj * tf.reduce_mean(tf.reduce_sum((1-tC0) * tf.squared_difference(tC0,pC0)+ (1-tC1) * tf.squared_difference(tC1,pC1), axis=[1,2,3]))
        #loss_c = tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(tc,pc)+ tC1 * tf.squared_difference(tc,pc), axis=[1,2,3]))
        loss_c = tf.reduce_mean(tf.reduce_sum(tC0 * tf.squared_difference(tc,pc), axis=[1,2,3]))
        tloss = loss_xy + loss_wh + loss_Cobj + loss_Cnoobj + loss_c
        return tloss 




class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.accuracy = 1000 

    def on_epoch_end(self, training_state):
        print("Loss:", training_state.val_loss) 
        if self.accuracy>training_state.val_loss:
           self.accuracy = training_state.val_loss 
           print("Model saved:", self.accuracy) 
           self.model.save('wyolo_yololite_dronesig.tfl')


#https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg
#this is yolo lite

network = tflearn.input_data(shape=[None,256,256,1],name="inp")
#1
network = tflearn.conv_2d(network, 16, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.layers.conv.max_pool_2d(network, 2,strides=2)
print(network.get_shape())
network = tflearn.conv_2d(network, 32, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.layers.conv.max_pool_2d(network, 2,strides=2)
print(network.get_shape())
network = tflearn.conv_2d(network, 64, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.layers.conv.max_pool_2d(network, 2,strides=2)
print(network.get_shape())
network = tflearn.conv_2d(network, 128, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.layers.conv.max_pool_2d(network, 2,strides=2)
print(network.get_shape())
network = tflearn.conv_2d(network, 128, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.layers.conv.max_pool_2d(network, 2,strides=2)
print(network.get_shape())
network = tflearn.conv_2d(network, 256, [3,3],activation='LeakyReLU')
print(network.get_shape())
network = tflearn.conv_2d(network, 125, [1,1],activation='linear') # linear
print(network.get_shape())


network = tflearn.fully_connected(network, grid_nums*grid_nums*(Ylastdim), activation='sigmoid',name="out")
print(network.get_shape())
network = tf.reshape(network,[-1,grid_nums,grid_nums,Ylastdim])
print(network.get_shape())
network = tflearn.regression(network, optimizer='adam',
                 #loss='mean_square',
                 loss=my_objective,
                 learning_rate=0.0001)

"""<h1>Option to Save Model Weights and History</h1>"""
model = tflearn.DNN(network,tensorboard_verbose=1, tensorboard_dir='./tflearn_logs/')
monitorCallback = MonitorCallback(model)


Train = True 
load_train = False 
if Train:
    model.fit(X_train, Y_train, n_epoch=64, shuffle=True,show_metric=False, 
              batch_size=128,validation_set=(X_test, Y_test), run_id='wyolo', callbacks=monitorCallback)
elif load_train:
    model.load('wyolo_yololite_dronesig.tfl')
    model.fit(X_train, Y_train, n_epoch=64, shuffle=True,show_metric=False, 
              batch_size=32,validation_set=(X_test, Y_test), run_id='wyolo', callbacks=monitorCallback)
else:
    model.load('wyolo_yololite_dronesig.tfl')
    

ev_loss=model.evaluate(X_test,Y_test,batch_size=32)
print("loss: ",ev_loss)


#Y_pred2 = model.predict(X_test[1000:1200])


