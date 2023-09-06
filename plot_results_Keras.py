#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 20:30:51 2020

@author: sanjoy
"""


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import math
try:
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Activation

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,recall_score,accuracy_score

from sklearn.metrics import confusion_matrix

import h5py


FFT_frames=256
FFT_N= 256



classes=["dx4e","dx6i","MTx","Nineeg","Parrot","q205","S500","tello","WiFi","wltoys"]
mods = len(classes)

Ylastdim=2*5+mods
maxsig=70
B = 2
C = len(classes)

iouval_comp=0.49


#dposl=dposl.reshape(dposl.shape[0],maxsig,dposl.shape[1])

shape = FFT_N
grid_nums=16
out_grid_shape=grid_nums
grid = shape/grid_nums
grid_val=grid

nfft=FFT_N


path = 'GenDataYOLO_10_snrtest.h5'
#path = '/home/sanjoybasak/Desktop/matlab_works/classify_yolo_sig/YOLO_mod_2/pycode_snrtest/GenDataYOLO_10_snrtest.h5'
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
print("num_sig_presence_test",num_sig_presence_test.shape)
print("--"*10)



def tf_iou(pxy0,pwh0,txy0,twh0):
    #iou calculation
    #8x8 constant
    #mulval = tf.constant([ 0,  32,  64,  96, 128, 160, 192, 224], dtype="float32")
    mulval = tf.constant([ 0,  32,  64,  96, 128, 160, 192, 224], dtype="float64")

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



    #https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy


def check_iousss(y_pred, y_true):
    pC0,pxy0,pwh0,pC1,pxy1,pwh1,pc = tf.split(y_pred, [1,2,2,1,2,2,mods], axis = 3)
    tC0,txy0,twh0,tC1,txy1,twh1,tc = tf.split(y_true, [1,2,2,1,2,2,mods], axis = 3)
        
    iou0 = tf_iou(pxy0,pwh0,txy0,twh0)
        #iou1 = tf_iou(pxy1,pwh1,txy1,twh1)


def get_IoU(pxy0,pwh0,txy0,twh0):
    px0=pxy0[0]; py0=pxy0[1]; pw0=pwh0[0]; ph0=pwh0[1]
    tx0=txy0[0]; ty0=txy0[1]; tw0=twh0[0]; th0=twh0[1]
    
    px0 = px0 * grid_val  
    py0 = py0 * grid_val 
    pw0 = pw0 * 256
    ph0 = ph0 * 256

    tx0 = tx0 * grid_val 
    ty0 = ty0 * grid_val 
    tw0 = tw0 * 256
    th0 = th0 * 256
    
    x11,y11,x12,y12 = px0-(pw0/2.0), py0-(ph0/2.0),px0+(pw0/2.0), py0+(ph0/2.0)
    x21,y21,x22,y22 = tx0-(tw0/2.0), ty0-(th0/2.0),tx0+(tw0/2.0), ty0+(th0/2.0)
    
    xm1 = np.maximum(x11,x21)
    ym1 = np.maximum(y11,y21)
    xm2 = np.minimum(x12,x22)
    ym2 = np.minimum(y12,y22)
    
    #intersection area
    ia = (xm2-xm1) * (ym2-ym1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - ia
    
    # compute the IoU
    iou = ia/ union_area
    iou = float("{:.2f}".format(iou))
    return iou
    
    


# function from some discussion
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        elif y_actual[i]==y_hat[i]==0:
           TN += 1
        elif y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


#rewrite this function  
def loss_separately(y_pred, y_true):
    lcord = 5 
    lnoobj =0.5
    #print('into loss func')
    
    
    #loss_c=np.zeros((y_pred.shape[0],1))
    
    
    # c,x,y,w,h,c,x,y,w,h  c =0,5 x= 
    pC0 = y_pred[:,:,:,0]
    pC1 = y_pred[:,:,:,5]
    pc  = y_pred[:,:,:,10:10+mods]
    pxy0= y_pred[:,:,:,1:2+1]
    pxy1= y_pred[:,:,:,6:7+1]
    pwh0= y_pred[:,:,:,3:4+1]
    pwh1= y_pred[:,:,:,8:9+1]
    
    tC0 = y_true[:,:,:,0]
    tC1 = y_true[:,:,:,5]
    tc  = y_true[:,:,:,10:10+mods]
    txy0= y_true[:,:,:,1:2+1]
    txy1= y_true[:,:,:,6:7+1]
    twh0= y_true[:,:,:,3:4+1]
    twh1= y_true[:,:,:,8:9+1]
    
    loss_xy =  np.mean(lcord *np.sum(tC0 * np.sum(np.square(txy0-pxy0),axis=3) + 
                                            tC1 * np.sum(np.square(txy1-pxy1),axis=3), axis=(1,2)))
    
    loss_wh =  np.mean(lcord *np.sum(tC0 * np.sum(np.square(np.sqrt(twh0)-np.sqrt(pwh0)), axis=3) + 
                                     tC1 * np.sum(np.square(np.sqrt(twh1) - np.sqrt(pwh1)),axis=3), axis=(1,2)))
    
    
    loss_Cobj = np.mean(np.sum(tC0 * np.square(tC0-pC0)+ tC1 * np.square(tC1-pC1), axis=(1,2)))
    loss_Cnoobj = lnoobj * np.mean(np.sum((1-tC0) * np.square(tC0-pC0)+ (1-tC1) * np.square(tC1-pC1), axis=(1,2)))
    
    loss_c=np.mean(np.sum(tC0 * np.sum(np.square(tc - pc),axis=3),axis=(1,2)))
     
    tloss = loss_xy + loss_wh + loss_Cobj + loss_Cnoobj + loss_c
    return loss_xy,loss_wh,loss_Cobj,loss_Cnoobj,loss_c,tloss




def get_results_YOLO(y_pred, y_true, threshold):
    
    pC0 = y_pred[:,:,:,0] #conf box 1
    pC1 = y_pred[:,:,:,5] #conf box 2
    pc  = y_pred[:,:,:,10:10+mods] #classification part 
    tC0 = y_true[:,:,:,0]
    tC1 = y_true[:,:,:,5]
    tc  = y_true[:,:,:,10:10+mods]
    
    #get xywh
    txy0= y_true[:,:,:,1:2+1]
    txy1= y_true[:,:,:,6:7+1]
    twh0= y_true[:,:,:,3:4+1]
    twh1= y_true[:,:,:,8:9+1]
    pxy0= y_pred[:,:,:,1:2+1]
    pxy1= y_pred[:,:,:,6:7+1]
    pwh0= y_pred[:,:,:,3:4+1]
    pwh1= y_pred[:,:,:,8:9+1]
    
    #check_iousss(y_pred, y_true)
    
    
    ## get TP, FP, TN, FN from tc and pc
    
    # step 1: prepare pc_temp from confidences of each boxes
    
    pc_reform=np.zeros((pc.shape[0],pc.shape[1],pc.shape[2],pc.shape[3]))
    
    P_Confidence_reform=np.zeros((pC0.shape[0],pC0.shape[1],pC0.shape[2]))
    
    for ii in range(P_Confidence_reform.shape[0]):
        b1_confid = np.argwhere(pC0[ii,:,:]>threshold) #compare conf of b1 with threshold
        b2_confid = np.argwhere(pC1[ii,:,:]>threshold) #compare conf of b2 with threshold
        for el in b1_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            if iouvalcur>iouval_comp:
                P_Confidence_reform[ii,el[0],el[1]]=1
                
        for el in b2_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            if iouvalcur>iouval_comp:
                P_Confidence_reform[ii,el[0],el[1]]=1
            
    
    
    for ii in range(pc.shape[0]):
        b1_pred = np.argwhere(P_Confidence_reform[ii,:,:]>threshold) #compare conf of b1 with threshold
        for el in b1_pred:
            clsmod=np.argmax(pc[ii,el[0],el[1],:])
            #check iou if >0.5 then put 1 (otherwise 0 is not needed as initiated with zeros)
            #iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            pc_reform[ii,el[0],el[1],clsmod]=1

    
    # step 2: compare tc and pc_reform to get TP, FP, TN, FN
    TP = FP = TN = FN = 0
    
    for ii in range(pc.shape[0]):
        for jj in range(pc.shape[1]):
            for kk in range(pc.shape[2]):
                y_pred_cur = pc_reform[ii,jj,kk,:] #pc_reform
                y_true_cur = tc[ii,jj,kk,:]
                TPc,FPc,TNc,FNc=perf_measure(y_true_cur,y_pred_cur)
                TP =TP + TPc
                FP =FP + FPc
                TN =TN + TNc
                FN =FN + FNc
                
    #print('Manual: TP, FP, TN, FN',TP, FP, TN, FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    #accuracy=(TP+TN)/(TP+TN+FP+FN)
    accuracy=(TP+TN)/(pc.shape[0]*pc.shape[1]*pc.shape[2]*pc.shape[3])
                
    
    # Find the probability of detection and False alarm rates
    
    TP =FP = TN = FN = 0
    
    for ii in range(pc_reform.shape[0]):
        for jj in range(pc_reform.shape[1]):
            y_pred_cur = P_Confidence_reform[ii,jj,:]
            y_true_cur = tC0[ii,jj,:]
            TPc,FPc,TNc,FNc=perf_measure(y_true_cur,y_pred_cur)
            TP =TP + TPc
            FP =FP + FPc
            TN =TN + TNc
            FN =FN + FNc
     
    #average class probabilities for box 1 and box 2    
    PD=TP/(TP+FN) #true positive ratio
    FA=FP/(FP+TN) #false positive ratio
    
    
    return precision,recall,accuracy,PD,FA


def plot_results_yolo(x_true,y_pred, y_true, threshold,snr):

    pC0 = y_pred[:,:,:,0] #conf box 1
    pC1 = y_pred[:,:,:,5] #conf box 2
    pc  = y_pred[:,:,:,10:10+mods] #classification part 
    tC0 = y_true[:,:,:,0]
    tC1 = y_true[:,:,:,5]
    tc  = y_true[:,:,:,10:10+mods]
    
    #get xywh
    txy0= y_true[:,:,:,1:2+1]
    txy1= y_true[:,:,:,6:7+1]
    twh0= y_true[:,:,:,3:4+1]
    twh1= y_true[:,:,:,8:9+1]
    pxy0= y_pred[:,:,:,1:2+1]
    pxy1= y_pred[:,:,:,6:7+1]
    pwh0= y_pred[:,:,:,3:4+1]
    pwh1= y_pred[:,:,:,8:9+1]
    
    #check_iousss(y_pred, y_true)
    
    
    ## get TP, FP, TN, FN from tc and pc
    
    # step 1: prepare pc_temp from confidences of each boxes
    
    pc_reform=np.zeros((pc.shape[0],pc.shape[1],pc.shape[2],pc.shape[3]))
    
    P_Confidence_reform=np.zeros((pC0.shape[0],pC0.shape[1],pC0.shape[2]))
    
    for ii in range(y_true.shape[0]):
        
        figs, axs = plt.subplots(nrows=1, ncols=2)
        ax=axs[0]
        ax.imshow(np.reshape(x_true[ii],[256,256]))
        
        #true plot
        b1 = np.argwhere(y_true[ii,:,:,0]>0.5)
        #plot true values on one subplot
        for el in b1:
            dval = y_true[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='k',facecolor='none')
            ax.add_patch(rect)
            box1_label_true=np.argmax(y_true[ii,el[0],el[1],10:10+mods])
            txt = classes[box1_label_true]
            ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
                                   fontsize=7, ha='center', va='center')
        
        b2 = np.argwhere(y_true[ii,:,:,5]>0.5)
        
        for el in b2:
            dval = y_true[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            box2_label_true=np.argmax(y_true[ii,el[0],el[1],10:10+mods])
            txt = classes[box2_label_true]
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
                                   fontsize=7, ha='center', va='center')
        
        
        #pred plot
        ax=axs[1]
        ax.imshow(np.reshape(x_true[ii],[256,256]))
        
        b1_confid = np.argwhere(pC0[ii,:,:]>threshold) #compare conf of b1 with threshold
        b2_confid = np.argwhere(pC1[ii,:,:]>threshold) #compare conf of b2 with threshold
        for el in b1_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            dval = y_pred[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='k',facecolor='none')
            ax.add_patch(rect)
            box1_label_true=np.argmax(y_pred[ii,el[0],el[1],10:10+mods])
            txt = classes[box1_label_true]
            txt=txt+','+str(iouvalcur)
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
                                   fontsize=10, ha='center', va='center')
            
                
        for el in b2_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            dval = y_pred[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            box2_label_true=np.argmax(y_pred[ii,el[0],el[1],10:10+mods])
            txt = classes[box2_label_true]
            txt=txt+','+str(iouvalcur)
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
                                   fontsize=10, ha='center', va='center')
            
            
        filename="results/snr_"+str(snr)+"_"+str(ii)+"_signum:"+str(1)+".png"
        plt.savefig(filename)
            

def plot_results_yolo_paper(x_true,y_pred, y_true, threshold,snr):

    pC0 = y_pred[:,:,:,0] #conf box 1
    pC1 = y_pred[:,:,:,5] #conf box 2
    pc  = y_pred[:,:,:,10:10+mods] #classification part 
    tC0 = y_true[:,:,:,0]
    tC1 = y_true[:,:,:,5]
    tc  = y_true[:,:,:,10:10+mods]
    
    #get xywh
    txy0= y_true[:,:,:,1:2+1]
    txy1= y_true[:,:,:,6:7+1]
    twh0= y_true[:,:,:,3:4+1]
    twh1= y_true[:,:,:,8:9+1]
    pxy0= y_pred[:,:,:,1:2+1]
    pxy1= y_pred[:,:,:,6:7+1]
    pwh0= y_pred[:,:,:,3:4+1]
    pwh1= y_pred[:,:,:,8:9+1]
    
    #check_iousss(y_pred, y_true)
    
    
    ## get TP, FP, TN, FN from tc and pc
    
    # step 1: prepare pc_temp from confidences of each boxes
    
    pc_reform=np.zeros((pc.shape[0],pc.shape[1],pc.shape[2],pc.shape[3]))
    
    P_Confidence_reform=np.zeros((pC0.shape[0],pC0.shape[1],pC0.shape[2]))
    
    for ii in range(y_true.shape[0]):
        
        figs, axs = plt.subplots(nrows=1, ncols=1,figsize=(14,14))
        #ax=axs[0]
        #ax.imshow(np.reshape(x_true[ii],[256,256]))
        
        #true plot
        b1 = np.argwhere(y_true[ii,:,:,0]>0.5)
        #plot true values on one subplot
        for el in b1:
            dval = y_true[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='k',facecolor='none')
            #ax.add_patch(rect)
            box1_label_true=np.argmax(y_true[ii,el[0],el[1],10:10+mods])
            txt = classes[box1_label_true]
            #ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
            #                       fontsize=7, ha='center', va='center')
        
        b2 = np.argwhere(y_true[ii,:,:,5]>0.5)
        
        for el in b2:
            dval = y_true[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=1,edgecolor='r',facecolor='none')
            #ax.add_patch(rect)
            box2_label_true=np.argmax(y_true[ii,el[0],el[1],10:10+mods])
            txt = classes[box2_label_true]
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            #ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
            #                       fontsize=7, ha='center', va='center')
        
        
        #pred plot
        #ax=axs[0]
        ax=axs
        ax.imshow(np.reshape(x_true[ii],[256,256]))
        
        b1_confid = np.argwhere(pC0[ii,:,:]>threshold) #compare conf of b1 with threshold
        b2_confid = np.argwhere(pC1[ii,:,:]>threshold) #compare conf of b2 with threshold
        for el in b1_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            dval = y_pred[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=4,edgecolor='k',facecolor='none')
            ax.add_patch(rect)
            box1_label_true=np.argmax(y_pred[ii,el[0],el[1],10:10+mods])
            txt2 = classes[box1_label_true]
            #txt=txt+','+str(iouvalcur)
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            #ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
            #                       fontsize=10, ha='center', va='center')
            
                
        for el in b2_confid:
            iouvalcur=get_IoU(pxy0[ii,el[0],el[1],:],pwh0[ii,el[0],el[1],:],txy0[ii,el[0],el[1],:],twh0[ii,el[0],el[1],:])
            dval = y_pred[ii,el[0],el[1]]
            rval = [(dval[1]*grid_val)+(grid_val*el[0]), (dval[2]*grid_val)+(grid_val*el[1]), dval[3]* 256, dval[4] * 256]
            rect = patches.Rectangle((rval[0]-rval[2]/2.0,rval[1]-rval[3]/2.0),rval[2],rval[3],linewidth=4,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            box2_label_true=np.argmax(y_pred[ii,el[0],el[1],10:10+mods])
            #txt = classes[box2_label_true]
            #txt=txt+','+str(iouvalcur)
            #txt = modnames[np.argmax(dval[10:])]+","+str(np.round(np.max(dval[10:]),2))
            #ax.annotate(txt, (rval[0], rval[1]), color='w', weight='bold', 
            #                       fontsize=10, ha='center', va='center')
            
        plt.show()    
        #filename="results/snr_"+str(snr)+"_"+str(ii)+"_sigT:"+txt+"P:"+txt2+".png"
        #plt.savefig(filename,dpi=100)


#https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg
#this is yolo lite

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
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=my_objective)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


checkpoint_path = "training_TF/cp.ckpt"
os.path.dirname(checkpoint_path)
cp_callback   = [
  EarlyStopping(monitor='val_loss', patience=20, mode='min'),
  ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, mode='min',verbose=1)]

model.load_weights(checkpoint_path)


ev_loss=model.evaluate(X_test,Y_test,batch_size=32)
print("loss: ",ev_loss)


plot_required=False
saveimage=False

idx=0

#mods=classes
snrs2 = np.unique(snrs_test)
num_sig_presence22=np.unique(num_sig_presence_test)

#sig_presenttt=6
idxx=0;

idx_4detprob=0

Detectionprobablity=np.zeros((len(snrs2),1))
Falsealarm=np.zeros((len(snrs2),1))

threshold=0.1
threshold_all=[0.1]
#accuracy_all=np.zeros((len(snrs2)*len(threshold_all)))
accuracy_all=np.zeros((len(snrs2),1))
precision_all=np.zeros((len(snrs2),1))
recall_all=np.zeros((len(snrs2),1))
f1_all=np.zeros((len(snrs2),1))
threshold_all=np.zeros((len(snrs2),1))
snr_all=np.zeros((len(snrs2),1))
PD_all=np.zeros((len(snrs2),1))
FA_all=np.zeros((len(snrs2),1))
loss_all=np.zeros((len(snrs2),1))
   
for snr in snrs2:
    #SNR_indxs=np.where((np.array(snrs_test)==snr) & (np.array(num_sig_presence_test)==7))
    SNR_indxs=np.where((np.array(snrs_test)==snr))
    #print(SNR_indxs)
    X_test_snr=X_test[SNR_indxs[0]]
    Y_test_snr=Y_test[SNR_indxs[0]]
    
    print(X_test_snr.shape)
    

    loss_curr=model.evaluate(X_test_snr,Y_test_snr,batch_size=32)        
    
    
    
    Y_predictions=model.predict(X_test_snr,batch_size=32)
    #plot_results_yolo(X_test_snr,Y_predictions,Y_test_snr,threshold,snr)
    plot_results_yolo_paper(X_test_snr,Y_predictions,Y_test_snr,threshold,snr)


    
    
    idxx+=1





