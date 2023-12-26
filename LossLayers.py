# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:35:25 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np

    
## MAKE LOSS: creates the model that calulates the loss and has outputs: 
 # loss, loss_val and outputs
 # This model has dummy input
def make_loss(u_model,n_modes,n_points,ablist,F_v,F_dv,VAL=False,n_pts_val=100,
              ERR=False,ab=[0,np.pi],n_pts_err=100,u_exact=[],du_exact=[],dtype='float64'):
    
    ##Takes inputs, constructs a function whose output is the loss. 
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    
    # The loss is the sum of local losses
    a,b = ablist[0]
    loss0 = loss_layer(u_model,n_modes,n_points,a,b,F_v,F_dv,
                           dtype=dtype)(xvals)
    for i in range(len(ablist)-1):
        a,b = ablist[i+1]
        # Local losses
        out_loss = loss_layer(u_model,n_modes,n_points,a,b,F_v,F_dv,
                               dtype=dtype)(xvals)
        loss0 += out_loss
    
    salida = [loss0]
    
    # If validation we have an extra layer for the validation
    if VAL:
        
        # The loss is the sum of local losses
        a,b = ablist[0]
        loss0_val = loss_layer(u_model,n_modes,n_pts_val,a,b,F_v,F_dv,
                               dtype=dtype)(xvals)
        for i in range(len(ablist)-1):
            a,b = ablist[i+1]
            # Local losses
            out_loss_val = loss_layer(u_model,n_modes,n_pts_val,a,b,F_v,F_dv,
                                   dtype=dtype)(xvals)
            loss0_val += out_loss_val
        
        salida.append(loss0_val)
        
    # If Error calculation 
    if ERR:
        error = error_layer(u_model,u_exact,du_exact,ab,n_pts_err,dtype=dtype)(xvals)
        salida.append(error)
    
    # The model that has outputs: loss, loss_val and error
    loss_model = tf.keras.Model(inputs=xvals,outputs=tf.stack(salida))
    
    return loss_model    


##Loss is evaluated in a non-trainable layer. The input has no effect
class loss_layer(tf.keras.layers.Layer):
    
    ## Inputs: Approximate solution, 
    ## Number of Fourier modes considered,
    ## Number of integration points,
    ## Interval (a,b) to calculate the loss
    
    ## Based on the weak formulation: int F_dv.v' + F_v.v dx = 0. 
    ## F_dv and F_v take arguments (u(x),u'(x),x)
    
    def __init__(self, u_model,n_modes,n_points,a,b,F_v,F_dv,dtype='float64',**kwargs):
        super(loss_layer, self).__init__()
        
        #Import solution model
        self.u_model=u_model
        
        #Generate integration points
        hi_pts = tf.convert_to_tensor(np.linspace(1,1+np.pi,n_points+1)-1, dtype=dtype)
        
        # The restriction of the quadrature rules to the subdomain
        filtered_tensora = tf.boolean_mask(hi_pts, hi_pts >= a)
        if not filtered_tensora[0] == a:
            filtered_tensora = tf.concat([[a],filtered_tensora], axis=0)   
        filtered_tensorb = tf.boolean_mask(filtered_tensora, filtered_tensora <= b)
        if not filtered_tensorb[-1] == b:
            filtered_tensorb = tf.concat([filtered_tensorb, [b]], axis=0)
        
        diff = tf.abs(filtered_tensorb[1:] - filtered_tensorb[:-1])
        self.pts = tf.constant(filtered_tensorb[:-1]+ diff/2,dtype=dtype)
        
        # Generate weights based on H^1_0 norm with no L2 component
        self.coeffs = np.array([((np.pi**2 * k**2) / (b - a)**2)**-0.5
                                for k in range(1, n_modes + 1)])

        ##----------
        # NOTE: The DST and DCT are computed explicitly here
        # To improve: Use the fast DST of python
        ##---------

        # Matrix for Sine transform
        V = np.sqrt(2./(b-a))
        DST = np.array([V*np.sin(np.pi*k*(self.pts-a)/(b-a))*diff
                                                 for k in range(1, n_modes+1)])

        # Matrix for Cosine transform
        self.DCT = np.array([V*(k*np.pi/(b-a))*np.cos(np.pi*k*(self.pts-a)/(b-a))*diff
                                                 for k in range(1, n_modes+1)])
        
        ##Import terms for weak formulation
        self.F_dv = F_dv 
        
        self.FT_low = tf.einsum("ji,i->j",DST,F_v(self.pts))
        
    def call(self,inputs):
        
        ## Evaluate u and its derivative at integration points
        ## Persistent True not necessary because it only evaluates u'(once)
        with tf.GradientTape() as t1:
            t1.watch(self.pts)
            u = self.u_model(self.pts)
        du = t1.gradient(u,self.pts)
        
        ##Evaluate terms for weak formulation
        high = self.F_dv(u,du,self.pts)
        
        ##Take appropriate transforms of each component
        FT_high = tf.einsum("ji,i->j",self.DCT,high)
        
        ##Add and multiply by weighting factors
        FT_tot = (FT_high+self.FT_low)*self.coeffs
        
        ##Return sum of squares loss 
        return  tf.reduce_sum(FT_tot**2)


##The H1 error is evaluated in a non-trainable layer. The input has no effect
# This function needs to be changed depending on the problem to solve 
# (e.g. singularities) and shoudl be tester for errors in H1 norm or semi-norm H1
class error_layer(tf.keras.layers.Layer):
    
    ##Inputs: approximate solution, 
    # Exact solution and derivative
    # Interval ab
    # points to evaluate the error
    
    def __init__(self, u_model,u_exact,du_exact,ab,n_points,dtype='float64',**kwargs):
        super(error_layer, self).__init__()
        
        self.u_model = u_model
    
        #Generate integration points
        hi_pts = np.linspace(1,1+np.pi,n_points+1)-1  
        diff = tf.abs(hi_pts[1:] - hi_pts[:-1])
        self.diff = tf.convert_to_tensor(diff, dtype=dtype)
        
        self.eval_pts = tf.constant(hi_pts[:-1]+ diff/2,dtype=dtype)
        
        #self.ue = u_exact(self.eval_pts)
        due = du_exact(self.eval_pts)
        self.due = tf.convert_to_tensor(due, dtype=dtype)
        
        # The H01 norm of the reference
        self.H1norm = tf.reduce_sum(due**2 * diff)
        
    def call(self,inputs):

        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.eval_pts)
            u  = self.u_model(self.eval_pts)
        du = t1.gradient(u,self.eval_pts)
        del t1
        
        H01norm = tf.reduce_sum((self.due - du)**2 * self.diff)
        
        ##Return the errors
        return (H01norm/self.H1norm)**(0.5)

