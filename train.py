import utils
import os
import pinn
import numpy as np
import tensorflow as tf

np.random.seed(1234)
tf.set_random_seed(1234)

##Training

P_back,x,y,P,rho,u,v,T,E,Et = utils.loadData('train',3)

print("Mother dataset (training) Properties:")
print("Mother dataset::: #Input Variables:  (P_back, x, y)")
print("Mother dataset::: #Output Variables: 5 (P, rho, u, v, T)")
print("Mother dataset::: Source: Fluent Simulation of max 15 iter/time_step")
print("Mother dataset::: Source: Residual Convergence Criteria: absolute 1e-06")
print("Mother dataset::: Method: Density based Explicit Solver with #Courant: 50")
print("Mother dataset::: Stagnation Pressure Inlet: 1.0 [atm], Static Pressure inlet = 1.0 [atm]")
print("Mother dataset::: Gauge Outlet Pressure Variation: [0.6* Po , 0.8 * Po]")
#print("Mother dataset::: #number nodes in Nozzle: " + str(n_values))
#print("Mother dataset::: Dataset #length for each variable: " + str(int(x_vector.shape[0])))


N_train = int(0.005 * P.shape[0])
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

layers = [3, 100, 100, 5]

model = pinn.PINN(P_back_train,\
                    x_train, \
                    y_train, \
                    P_train, \
                    rho_train, \
                    u_train, \
                    v_train, \
                    T_train, \
                    layers)


#model.save(os.getcwd() + '/save/')


#Load trained model
saver = tf.compat.v1.train.import_meta_graph('tmp/2d_inviscid_model-20000.meta')
saver.restore(model.sess,tf.train.latest_checkpoint('tmp/.'))

loss_vector = model.train(40001)


'''
##Testing 
P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,T_test,E_test,Et_test = utils.loadData('test',3)

P_back_test = P_back_test.flatten()[:,None]
x_test = x_test.flatten()[:,None]
y_test = y_test.flatten()[:,None]

#Prediction
P_pred, rho_pred, u_pred, v_pred, T_pred = model.predict(P_back_test, x_test,y_test)




#Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))


error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_T = np.linalg.norm(T_test-T_pred,2)/np.linalg.norm(T_test,2)
print("Test Error in T: "+str(error_T))
'''