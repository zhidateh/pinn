import csv
import numpy as np


P=[]
rho=[]
E=[]
u=[]

with open('cdn_P.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            P.append(float(e))

with open('cdn_rho.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            rho.append(float(e))

with open('cdn_E.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            E.append(float(e))

with open('cdn_u.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            u.append(float(e))

P = np.asarray(P)
rho = np.asarray(rho)
E = np.asarray(E)
u = np.asarray(u)

pb=[]
pb=np.asarray(pb)

z=[]
z=np.asarray(z)

for i in range(0, 27):
    P_back = 0.01*(21+3*i)*np.ones((101,1))
    pb = np.concatenate((pb, P_back), axis=None)
    x_l = 0.01*np.arange(0, 101, dtype=float).flatten()[:,None]
    z = np.concatenate((z, x_l), axis=None)

train_frac = 0.01
N_train = int(train_frac*P.shape[0])

A = np.random.choice(range(P.shape[0]), size=(N_train,), replace=False)

# x
P_back_train = pb[A].flatten()[:,None]
x_train = z[A].flatten()[:,None]
# y
P_train = P[A].flatten()[:,None]
rho_train = rho[A].flatten()[:,None]
u_train = u[A].flatten()[:,None]
E_train = E[A].flatten()[:,None]

print("N train is %d\n"%N_train)

print("A is ",A,"\n")

print(P_back_train.shape)

