# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:33:16 2023

@author: jamie.taylor
"""

import numpy as np
import tensorflow as tf

from SRC.LossLayers import make_loss
from SRC.LossLayers_LS import make_least_squares_loss 


##Loss function for training
def tricky_loss(y_true,y_pred):
    return y_pred[0]

##Loss function for validation
def val_loss(y_true,y_pred):
    return y_pred[1]

##Error H1 or H01 (if present is in pos -1)
def sol_err(y_true, y_pred):
    return y_pred[-1]


##constructs the loss model and trains the model. 
# u_model,n_modes,n_points,ablist,F_v,F_dv,epochs,lr : All explained in main 
# Validation is by default False
def train_iteration(u_model,n_modes,n_points,ablist,F_v,F_dv,epochs,lr,
                    nn_last,lam=10**-5,
                    VAL=False,n_pts_val=100,
                    ERR=False,ab=[0,np.pi],n_pts_error=100,u_exact=[],du_exact=[],dtype='float64'):
    
    # Create the big model with las layer = loss
    loss_model = make_loss(u_model,n_modes,n_points,ablist,F_v,F_dv,VAL,n_pts_val,
                          ERR,ab,n_pts_error,u_exact,du_exact,dtype=dtype)
    
    # If least squares we wrap the model in other one
    min_model = make_least_squares_loss(loss_model,nn_last=nn_last,lam=lam,dtype=dtype)
    
    # Adam is our best optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    metrics = []
    callback = []
    # If validation is True we also report the loss validation at each iteration
    if VAL: 
        metrics.append(val_loss)
    # If errror is True we also report the error H^1 and H_0^1 at each iteration
    if ERR: 
        metrics.append(sol_err)
    
    min_model.compile(optimizer=optimizer,loss= tricky_loss, metrics = metrics)
    
    # Entrenamiento
    history = min_model.fit(x=tf.constant([1.]),y=tf.constant([1.]),
                            epochs=epochs, callbacks=[callback])
    
    return history


   

    

