# -*- coding: utf-8 -*-
'''
Created on Mon May 22 12:33:07 2023

@author: jamie.taylor
'''

import tensorflow as tf
import numpy as np

from SRC.Architecture import make_u_linear_model
from SRC.Training import train_iteration
from SRC.RefineTest import ref_identify


# ========================================
# RANDOM NUMBERS AND PRECISION SETTING

tf.random.set_seed(1234)
np.random.seed(1234)
# Set the random seed
tf.keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
tf.keras.backend.set_floatx(dtype)


# ========================================
# Solve the problem # (folder problems )
from Problem1 import problem

# Weak formulation is int F_dv.v' + F_v.v dx. 
F_dv = problem['Fdv'] #La notacion de Fv Fdv corresponde con el paper DFR 
F_v = problem['Fv']

# Solucion y derivada (se usan solo cuando se va a calcular error (training+validation))
du_exact = problem['der_exact']
u_exact = problem['exact']


# ========================================
# Parametros del entrenamiento

# Numero de modos de fourier (funciones base)
n_modes = 5
# Numero de puntos de integracion (DST/DCT)
n_points = 500

# Numero de refinamientos (+1)
# n_ref = 1 do the training only once! (NO REFINEMENT!)
n_ref = 6

# Numero de iteraciones por ciclo de refinamiento 
iterations = 500
# Learning rate (Adam)
lr = 10**(-3.5) 

# Numero de capas, neuronas en cada capa y neuronas de la ultima capa
# make_u_linear model is replaced here when LS implementation
n_layers = 3 #(Hidden)
nn = 10 #number of neurons on all the layers except last
nn_last = 20

u_model = make_u_linear_model(nn,nn_last,n_layers,activation='tanh',train=False,dtype=dtype)

# Incluir validacion o no
VAL = True
n_pts_val = round(2.17*n_points) #Number of points to use for validation

# Incluir calculo del error en cada iteracion o no
# OJO: El calculo adecuado del error depende del problema : 
    # modificar LossLayers.py-error_layer
ERR = True
n_pts_error = 5000 #Number of points in the interval [a,b] to calculate the error

# Intevalos iniciales
# ab_list corresponde a la lista de subdominios 
vertices = [[0+j*np.pi/4 for j in range(5)]]
ab_list = [[vertices[0][j],vertices[0][j+2]] for j in range(3)]

ablist_all = [ab_list]
n_subdomains = [len(ab_list)]

# From here ... the magic... 
# ========================================

total_loss = [None] * (n_ref)

# Saving values of validation and error
if VAL: 
    total_loss_val = [None]*n_ref
    
if ERR:
    total_error = [None]*n_ref

## Lop of refinements 
elapsed = []
for i in range(n_ref):
    
    # ========================================
    # ENTRENAMIENTO 
    history = train_iteration(u_model,n_modes,n_points,ab_list,F_v,F_dv,
                              epochs = iterations, lr = lr,
                              nn_last=nn_last,lam=10**-5,
                              VAL=VAL,n_pts_val=n_pts_val,
                              ERR=ERR,ab=[0,np.pi],n_pts_error=n_pts_error,
                              u_exact=u_exact,du_exact=du_exact,dtype=dtype)
    
    # ========================================
    # Save all values of the losses and error
    total_loss[i] = np.array(history.history['loss'])
    if VAL:
        total_loss_val[i] = np.array(history.history['val_loss'])
    if ERR:
        total_error[i] = np.array(history.history['sol_err'])
    
    # ========================================
    # Refinamiento de los subdominios
    if i < n_ref-1: #The last one is not necessary
        ab_list,new_vert = ref_identify(u_model,F_v,F_dv,u_exact,n_modes,n_points,
                                        ab_list,dtype)
        vertices.append(new_vert)
        ablist_all.append(ab_list)
    
        print('\n Total number of subdomains: %.i\n' %len(ab_list))
        n_subdomains.append(len(ab_list))
    
    