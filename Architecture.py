# -*- coding: utf-8 -*-
'''
Created on Mon May 22 12:35:13 2023

@author: jamie.taylor
'''

import tensorflow as tf
import numpy as np

## Fully connected feed forward network with nn neurons per layer
## and l hidden layers. Cutoff function applied at the end for BC. 
# The last layer is a customized linear layer (WITHOUT BIAS)

def make_u_linear_model(nn,nn_last,n_layers,activation,train,dtype='float64'):
    
    # If we will use LS the input train is False (Not training the parameters of the last layer)
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    
    l1 = tf.keras.layers.Dense(nn,activation=activation)(xvals)
    for i in range(n_layers-2):
        l1 = tf.keras.layers.Dense(nn,activation=activation,dtype=dtype)(l1)
    
    # Penultima layer(output nn_last)
    l0 = tf.keras.layers.Dense(nn_last,activation=activation,dtype=dtype)(l1)
    # Last layer is cutomize
    l2 = linear_last_layer(nn_last,train=train,dtype=dtype)(l0)
    
    # Imposing the B.C homogeneous
    output = cutoff_layer_linear()([xvals,l2])
    
    u_model = tf.keras.Model(inputs = xvals,outputs = output)
    
    u_model.summary()
    
    return u_model
    
# The last layer (customized)
class linear_last_layer(tf.keras.layers.Layer):
    def __init__(self,nn_last,train,dtype='float64',**kwargs):
        super(linear_last_layer,self).__init__()
        
        # Standard initialization of the weights (TF)
        pweight = tf.random.uniform([nn_last],minval=-(6/(1+nn_last))**0.5,
                                    maxval=(6/(1+nn_last))**0.5, dtype =dtype)
        #pb = tf.constant([0.],dtype = dtype)
        #self.vars = tf.Variable(tf.concat([pweight,pb],axis=-1),trainable=train,dtype =dtype)
        self.vars = tf.Variable(pweight,trainable=train,dtype=dtype)
    
    def call(self,inputs):
        pweights = self.vars #[:-1]
        #bias = self.vars[-1]
        
        return tf.einsum("i,ji->j",pweights,inputs)#+bias

# The cutoff function applied to the last layer 
class cutoff_layer_linear(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(cutoff_layer_linear,self).__init__()
    
    def call(self,inputs):
        x,u=inputs
        # cut = (x-2)*(x-5)
        cut = x*(x-np.pi)
        return tf.einsum("ij,i->i",cut,u)   
    
    

## ====================================================== 
# Functions implemented as standart NN for the u_model 
## ====================================================== 

##Fully connected feed forward network with nn neurons per layer
## and l hidden layers. Cutoff function applied at the end for BC. 
def make_u_model(nn,nl,activation='tanh',dtype='float64'):
    
    xvals = tf.keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)
    
    # Dense layer 
    l1 = tf.keras.layers.Dense(nn,activation=activation,dtype=dtype)(xvals)
    for i in range(nl-2):
        l1 = tf.keras.layers.Dense(nn,activation=activation,dtype=dtype)(l1)
    l2=tf.keras.layers.Dense(1)(l1)
    
    # The outputs are the neurons (basis functions)
    output = cutoff_layer()([xvals,l2])
    u_model = tf.keras.Model(inputs = xvals,outputs = output)
    
    return u_model

## layer to implement the zero dirichlet boundary condition via cutoff
class cutoff_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(cutoff_layer,self).__init__()
    
    def call(self,inputs):
        x,u = inputs
        # cut = (x-2)*(x-5)
        cut = x*(x-np.pi)
        return tf.einsum('ij,ij->i',cut,u)
    
    