# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:35:25 2023

@author: jamie.taylor
"""

import tensorflow as tf


# Makes the LS loss model
# This function wraps the model of the loss to compute the LS solver of 
    # the last layer  
def make_least_squares_loss(loss_model,nn_last,lam=10**-5,dtype='float64'):
    
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    loss = ls_loss_layer(loss_model,nn_last,lam=lam,dtype=dtype)(xvals)
   
    out_model = tf.keras.Model(inputs=xvals,outputs=loss)
    
    out_model.summary()
    
    return out_model 


# This layer computes the LS solver of the last layer of the loss_model
class ls_loss_layer(tf.keras.layers.Layer):
    
    def __init__(self,loss_model,nn_last,lam,dtype,**kwargs):
        super(ls_loss_layer, self).__init__()
        self.loss_model = loss_model
        self.nn_last = nn_last
        self.lam = lam
        self.dtype0 = dtype
        
    def call(self,inputs):
        loss = least_squares_fn(self.loss_model,self.nn_last,self.lam,dtype=self.dtype0)
        return loss

# The function that actually compiute the weights 
# OJO! The weights are always assign to 0 for the sake of comparison
def least_squares_fn(loss_model,nn,lam=10**-5,dtype='float64'):
    
    # Assign the weights to be 0
    loss_model.layers[1].u_model.layers[-2].vars.assign(tf.zeros([nn],dtype=dtype))
    
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(loss_model.layers[1].u_model.layers[-2].vars)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(loss_model.layers[1].u_model.layers[-2].vars)
            # The output of the loss model has 4 outputs if VAL and ERR
            outputloss = loss_model(tf.constant([1.]))
            l = outputloss[0]
        # This method implies computing two derivatives
        B = t2.gradient(l,loss_model.layers[1].u_model.layers[-2].vars)
    A = t1.jacobian(B, loss_model.layers[1].u_model.layers[-2].vars)
    
    regul = tf.eye(nn,nn,dtype=dtype)*lam
    
    w_new = tf.reshape(tf.linalg.solve(A+regul,-tf.reshape(B,[nn,1])),[nn])
    
    loss_model.layers[1].u_model.layers[-2].vars.assign(w_new)
    
    return loss_model(tf.constant([1.])) 
    