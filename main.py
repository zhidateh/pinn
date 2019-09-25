import init
import os
import pinn
import numpy as np
import tensorflow as tf

np.random.seed(1234)
tf.set_random_seed(1234)

wd = os.getcwd() + '/case/'
P_back,x,y,P,rho,u,v,T,E,Et = init.loadData(wd)


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


N_train = int(0.01 * P.shape[0])
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

layers = [3, 10, 10, 10, 5]

model = pinn.PINN(P_back_train,\
                    x_train, \
                    y_train, \
                    P_train, \
                    rho_train, \
                    u_train, \
                    v_train, \
                    T_train, \
                    layers)

model.train(10)

print("hi")