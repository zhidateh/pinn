#!/usr/bin/python3

import utils as utils
import os
import pinn as pinn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math

np.random.seed(1234)
tf.set_random_seed(1234)

setting_name = "PINN00100"
node_filter = 1
layers = [3, 10, 10, 10,  5]

num_epochs      = 10
num_iter        = 5001
batch_size      = 1000  
learning_rate   = 1e-3

test_path = '/home/zhida/Documents/PINN/test/'
train_path = '/home/zhida/Documents/PINN/train/'

##Load all data---------------------------------------------------------------------------------------  
#data is matrix of m x n, where m is number of nodes, n is number of datasets
P_back,x,y,P,rho,u,v,T,Et,E = utils.loadData(train_path,node_filter)
P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,T_test,Et_test,E_test = utils.loadData(test_path,node_filter)


##Prepocessing---------------------------------------------------------------------------------------
#normalize data with stagnation condition
P_norm      = np.amax(P)
rho_norm    = np.amax(rho)
u_norm      = np.amax(u)
v_norm      = np.amax(v)
T_norm      = np.amax(T)
Et_norm     = np.amax(Et)
E_norm      = np.amax(E)

P_back      /= P_norm
P           /= P_norm
rho         /= rho_norm
u           /= u_norm 
v           /= v_norm
T           /= T_norm
Et          /= Et_norm
E           /= E_norm

P_back_test /= P_norm
P_test      /= P_norm
rho_test    /= rho_norm
u_test      /= u_norm
v_test      /= v_norm
T_test      /= T_norm
Et_test     /= Et_norm
E_test      /= E_norm


P_back  = P_back.flatten()[:,None]
x       = x.flatten()[:,None]
y       = y.flatten()[:,None]
P       = P.flatten()[:,None]
rho     = rho.flatten()[:,None]
u       = u.flatten()[:,None]
v       = v.flatten()[:,None]
T       = T.flatten()[:,None]
Et      = Et.flatten()[:,None]

P_back_test = P_back_test.flatten()[:,None]
x_test = x_test.flatten()[:,None]
y_test = y_test.flatten()[:,None]
P_test = P_test.flatten()[:,None] 
rho_test = rho_test.flatten()[:,None]
u_test = u_test.flatten()[:,None]
v_test = v_test.flatten()[:,None]
T_test = T_test.flatten()[:,None]
E_test = E_test.flatten()[:,None]
Et_test = Et_test.flatten()[:,None]



#Main code-------------------------------------------------------------------------------
#initiaise PINN class
model = pinn.DeepPINN_2D(P_back,\
                        x, \
                        y, \
                        P, \
                        rho, \
                        u, \
                        v, \
                        Et, \
                        layers)


model.ckpt_name = 'tmp/' + setting_name 

#saver = tf.compat.v1.train.import_meta_graph('/home/zhida/Documents/pinn/2d_inviscid_model/PINN00037/PINN00037-19.meta')
#saver.restore(model.sess,("/home/zhida/Documents/pinn/2d_inviscid_model/PINN00037/PINN00037-19"))
model.train(num_epochs, num_iter, batch_size,learning_rate)

# #Prediction----------------------------------------------------------------------------------------------------------------
P_pred, rho_pred, u_pred, v_pred, Et_pred = model.predict(P_back_test, x_test,y_test)

# #Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_Et = np.linalg.norm(Et_test-Et_pred,2)/np.linalg.norm(Et_test,2)
print("Test Error in E: "+str(error_Et))

P_pred       *= P_norm
rho_pred     *= rho_norm
u_pred       *= u_norm
v_pred       *= v_norm
Et_pred       *= Et_norm

path = os.getcwd() + '/predict/%s_bp=%s.csv'%(setting_name,str(int(P_back_test[0]*P_norm)))
utils.writeData(path,x_test,y_test,P_pred,rho_pred,u_pred,v_pred,Et_pred)

path2 = os.getcwd() + '/predict/%s_bp=%s_loss.csv'%(setting_name,str(int(P_back_test[0]*P_norm)))
utils.writeLoss(path2,model.loss_vector,model.step_vector)

