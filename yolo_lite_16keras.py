#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:10:30 2020

@author: sbasak
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
try:
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#import matplotlib.pyplot as plt
#import matplotlib.patches as patches

import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib
#import tflearn

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




def my_objective(y_true, y_pred):
    lcord = 5 
    lnoobj =0.5
    pC0,pxy0,pwh0,pC1,pxy1,pwh1,pc = tf.split(y_pred, [1,2,2,1,2,2,mods], axis = 3)
    tC0,txy0,twh0,tC1,txy1,twh1,tc = tf.split(y_true, [1,2,2,1,2,2,mods], axis = 3)
    
    loss_xy = lcord * tf.reduce_mean(tf.reduce_sum(tC0 * tf.math.squared_difference(txy0,pxy0)+ tC1 * tf.math.squared_difference(txy1,pxy1), axis=[1,2,3]))
    loss_wh = lcord * tf.reduce_mean(tf.reduce_sum(tC0 * tf.math.squared_difference(tf.sqrt(twh0),tf.sqrt(pwh0))+ tC1 * tf.math.squared_difference(tf.sqrt(twh1),tf.sqrt(pwh1)), axis=[1,2,3]))
    loss_Cobj = tf.reduce_mean(tf.reduce_sum(tC0 * tf.math.squared_difference(tC0,pC0)+ tC1 * tf.math.squared_difference(tC1,pC1), axis=[1,2,3]))
    loss_Cnoobj = lnoobj * tf.reduce_mean(tf.reduce_sum((1-tC0) * tf.math.squared_difference(tC0,pC0)+ (1-tC1) * tf.math.squared_difference(tC1,pC1), axis=[1,2,3]))
    loss_c = tf.reduce_mean(tf.reduce_sum(tC0 * tf.math.squared_difference(tc,pc), axis=[1,2,3]))
    tloss = loss_xy + loss_wh + loss_Cobj + loss_Cnoobj + loss_c
    return tloss 


NetworkInput = Input(shape=[256,256,1],name="inp")
network = Conv2D(16,(3,3),padding='same')(NetworkInput)
network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)

network = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(network)


network = Conv2D(32,(3,3),padding='same')(network)
network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)
network = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(network)

network = Conv2D(64,(3,3),padding='same')(network) 
network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)
network = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(network)

network = Conv2D(128,(3,3),padding='same')(network) 
network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)
network = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(network)


network = Conv2D(128,(3,3),padding='same')(network) 
network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)
network = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(network)

network = Conv2D(256,[3,3],padding='same')(network) 

network = tf.keras.layers.LeakyReLU(alpha=0.01)(network)

network = Conv2D(125,(1,1),padding='same',activation='linear')(network)



network = Flatten()(network)

network = Dense(grid_nums*grid_nums*(Ylastdim) , activation='sigmoid')(network)

network = tf.reshape(network,[-1,grid_nums,grid_nums,Ylastdim])

model = Model(inputs = NetworkInput, outputs = network)
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=my_objective)
model.summary()


checkpoint_path = "training_TF/cp.ckpt"
os.path.dirname(checkpoint_path)
cp_callback   = [
  EarlyStopping(monitor='val_loss', patience=20, mode='min'),
  ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, mode='min',verbose=1)]


Train = True 
load_train = False 
if Train:
    history = model.fit(X_train, Y_train, epochs = 200, batch_size = 128,
                    callbacks=[cp_callback], validation_data=(X_test, Y_test))

elif load_train:
    model.load_weights(checkpoint_path)
    history=model.fit(X_train, Y_train, epochs = 10, batch_size = 64,
                    callbacks=[cp_callback], validation_data=(X_test, Y_test))
 


model.load_weights(checkpoint_path)
ev_loss=model.evaluate(X_test,Y_test,batch_size=32)
print("loss: ",ev_loss)
       


