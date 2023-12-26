# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:53:13 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np

def u_exact(x):
    return x*(x-np.pi)*tf.math.exp(-120*(x-np.pi/2)**2)

# Derivative of u_exact
def du_exact(x):
    with tf.GradientTape() as t1:
        t1.watch(x)
        ue = u_exact(x)
    due = t1.gradient(ue,x)
    return due
        
    #return -tf.math.exp(-30*(np.pi-2*x)**2)*(np.pi- 2*x + 120*np.pi**2*x - 360*np.pi*x**2 + 240*x**3)

def F_v(x):
    with tf.GradientTape() as t1:
        t1.watch(x)
        with tf.GradientTape() as t2:
            t2.watch(x)
            ue = u_exact(x)
        due = t2.gradient(ue,x)
    ddue = t1.gradient(due,x)
    return ddue

def F_dv(u,du,x):
    return du


##Problem consists of a dictionary with the terms F_v,F_dv and exact solution
problem = {"exact" : u_exact,
           'der_exact' : du_exact,
           "Fdv" : F_dv,
           "Fv" : F_v,
           }

