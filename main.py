import utils
import os
import pinn2 as pinn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.set_random_seed(1234)

node_filter = 1
dataset_filter = 0.005

#layers = [3, 100,200,300, 200, 100, 5]
layers = [3, 50, 100 , 50, 5]
#layers = [3, 50, 50, 50, 50, 50, 50, 50, 5]

training_itr = 20001


##Load training data---------------------------------------------------------------------------------------

P_back,x,y,P,rho,u,v,T,E,Et = utils.loadData('train',node_filter)

# print("Mother dataset (training) Properties:")
# print("Mother dataset::: #Input Variables:  (P_back, x, y)")
# print("Mother dataset::: #Output Variables: 5 (P, rho, u, v, T)")
# print("Mother dataset::: Source: Fluent Simulation of max 15 iter/time_step")
# print("Mother dataset::: Source: Residual Convergence Criteria: absolute 1e-06")
# print("Mother dataset::: Method: Density based Explicit Solver with #Courant: 50")
# print("Mother dataset::: Stagnation Pressure Inlet: 1.0 [atm], Static Pressure inlet = 1.0 [atm]")
# print("Mother dataset::: Gauge Outlet Pressure Variation: [0.6* Po , 0.8 * Po]")
# print("Mother dataset::: #number nodes in Nozzle: " + str(n_values))
# print("Mother dataset::: Dataset #length for each variable: " + str(int(x_vector.shape[0])))


N_train = int(dataset_filter * P.shape[0])
A = np.random.choice(P_back.shape[0],size = (N_train,), replace=False)


#training data
P_back_train    = P_back[A].flatten()[:,None]
x_train         = x[A].flatten()[:,None]
y_train         = y[A].flatten()[:,None]
P_train         = P[A].flatten()[:,None]
rho_train       = rho[A].flatten()[:,None]
u_train         = u[A].flatten()[:,None]
v_train         = v[A].flatten()[:,None]
T_train         = T[A].flatten()[:,None]
E_train         = E[A].flatten()[:,None]
Et_train        = Et[A].flatten()[:,None]





plt.plot(x_train,y_train,'ro')
plt.show()




model = pinn.PINN(P_back_train,\
                    x_train, \
                    y_train, \
                    P_train, \
                    rho_train, \
                    u_train, \
                    v_train, \
                    E_train, \
                    layers)




#Load trained model-------------------------------------------------------------------------------------------------
#saver = tf.compat.v1.train.import_meta_graph('tmp/2d_inviscid_model_20190929-193536-50000.meta')
#saver.restore(model.sess,tf.train.latest_checkpoint('tmp/.'))
#--------------------------------------------------------------------------------------------------------------------




#Training-----------------------------------------------------------------------------------------------------------
loss_vector = model.train(training_itr)
#model.save(os.getcwd() + '/save/')
#-------------------------------------------------------------------------------------------------------------------





##Testing---------------------------------------------------------------------------------------------------------------
P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,T_test,E_test,Et_test = utils.loadData('test',node_filter)

P_back_test = P_back_test.flatten()[:,None]
x_test = x_test.flatten()[:,None]
y_test = y_test.flatten()[:,None]

#Prediction
P_pred, rho_pred, u_pred, v_pred, E_pred = model.predict(P_back_test, x_test,y_test)

#model.load(os.getcwd() + '/save/')
#P_pred_1, rho_pred, u_pred, v_pred, T_pred = model.predict(P_back_test, x_test,y_test)

plt.plot(np.linspace(0,1,len(loss_vector)), loss_vector,'rx')
plt.show()


#print(P_pred_1)

utils.writeData(x_test,y_test,P_pred,rho_pred,u_pred,v_pred,E_pred)


#Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))


error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_T = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
print("Test Error in T: "+str(error_T))
#------------------------------------------------------------------------------------------------------------------------
