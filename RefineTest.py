# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:34:27 2023

@author: jamie.taylor
"""


import tensorflow as tf
from itertools import chain

from SRC.LossLayers import loss_layer


# Funcion que evalua (nuevamente pero fuera del entrenamiento)
# el loss en cada subdominio
def test_refine(F_v,F_dv,ablist,u_model,n_modes,n_points,dtype='float64'):
    
    a,b = ablist[0]
    res_vec0 = loss_layer(u_model,n_modes,n_points,a,b,F_v,F_dv,
                           dtype=dtype)(tf.constant(1.))
    list_res = [res_vec0]
    for i in range(len(ablist)-1):
        a,b = ablist[i+1]
        # Local losses
        out_loss = loss_layer(u_model,n_modes,n_points,a,b,F_v,F_dv,
                               dtype=dtype)(tf.constant(1.))
        list_res.append(out_loss)
      
    return list_res

# Funcion que refina la malla basado en marcar y refinar con dos subelementos
# este proceso corresponde al reportado en el articulo: Adaptive DFR
def ref_identify(u_model,F_v,F_dv,u_exact,n_modes,n_points,ab_list,dtype='float64'):
    
    # Create an uniform refinement of each of the subdomains
    div_ab = []      
    for i in range(len(ab_list)):
        new_pts = [ab_list[i][0]+j*(-ab_list[i][0]+ab_list[i][1])/4 for j in range(5)]
        div_ab += [[new_pts[j],new_pts[j+2]] for j in range(3)]
    
    # Include only the subdomains not repeated
    unif_ab = []
    for sublist in div_ab:
        if sublist not in unif_ab:
            unif_ab.append(sublist)
    
    # =============================================================================
    # The evaluated loss - error indicator
    # =============================================================================
    losslist = test_refine(F_v,F_dv,unif_ab,u_model,n_modes,n_points,dtype)
    
    # Calculate the threshold value
    threshold = 0.66*max(losslist)#tf.math.reduce_mean(losslist)
      
    # Create a new list with elements from ablist that are not in positions_to_delete
    new_ab = ab_list.copy()
    new_ab0 = [unif_ab[i] for i in range(len(unif_ab)) if losslist[i] > threshold]
    
    # Include the new subdomains
    for sublist in new_ab0:
        if sublist not in new_ab:
            new_ab.append(sublist)
            
    new_vert = list(set(chain.from_iterable(new_ab)))
    
    return new_ab, new_vert

