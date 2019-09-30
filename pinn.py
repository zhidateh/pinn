import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


class PINN:
    # Initialize the class
    def __init__(self, P_back, x, y, P, rho, u, v, T, layers):
        
        self.chkpt_name = 'tmp/2d_inviscid_model_%s'%time.strftime("%Y%m%d-%H%M%S")



        #P_back     : m x 1 matrix
        #x          : m x 1 matrix
        #y          : m x 1 matrix
        
        #X          : m x 3 matrix
        X = np.concatenate([P_back, x, y], 1)

        #Find minimum of each column
        self.lb = X.min(0)

        #Find maximum of each column
        self.ub = X.max(0)
        
        self.X      = X
        self.P_back = P_back
        self.x      = x
        self.y      = y
        self.P      = P
        self.rho    = rho
        self.u      = u
        self.v      = v
        self.T      = T

        # e.g.       : layers = [2, 10, 10, 10, 4]
        # no of input           : 2
        # no of output          : 4
        # no of hidden layers   : 3
        # no of neurons         : 10
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                        log_device_placement=True))
        
        self.P_back_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.P_back.shape[1]])
        self.x_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.P_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.P.shape[1]])
        self.rho_tf     = tf.compat.v1.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.T_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, self.T.shape[1]])
        
        self.P_pred, self.rho_pred, self.u_pred,self.v_pred, self.T_pred, self.e_res = self.net_NS(self.P_back_tf, self.x_tf,self.y_tf)

        # MSE Normalization
        P_norm      = np.amax(P)
        rho_norm    = np.amax(rho)
        u_norm      = np.amax(u)
        v_norm      = np.amax(v)
        T_norm      = np.amax(T)
        e_norm      = 1#np.amax(P/y)

        ##have to change here------------------------------------------------------------------
        a = 2

        self.loss = tf.reduce_mean(tf.square(self.P_tf - self.P_pred))/(P_norm**2) + \
                    tf.reduce_mean(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2) + \
                    tf.reduce_mean(tf.square(self.u_tf - self.u_pred))/(u_norm**2) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.u_pred))/(v_norm**2) + \
                    tf.reduce_mean(tf.square(self.T_tf - self.T_pred))/(T_norm**2) + \
                    a*tf.reduce_mean(tf.square(self.e_res))/(e_norm**2)
                 
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 15000,
                                                                           'maxfun': 15000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, P_back, x, y):
        P_rho_u_v_T = self.neural_net(tf.concat([P_back, x, y], 1), self.weights, self.biases)
        P   = P_rho_u_v_T[:,0:1]
        rho = P_rho_u_v_T[:,1:2]
        u   = P_rho_u_v_T[:,2:3]
        v   = P_rho_u_v_T[:,3:4]
        T   = P_rho_u_v_T[:,4:5]

        ##have to change here------------------------------------------------------------------
        mu = 0.0
        R = 287
        gamma = 1.4
        k = 0.0242
        E = R*T/(gamma-1) + (u*u+v*v)/2

        H_v2 = - 2/3*y*tf.gradients(u*v/y, x)[0]
        H_v3 = - y*2/3*tf.gradients(u*v/y, y)[0]
        H_v4 = k*tf.gradients(T, y)[0] 

        #     E1        E2              E3          E4
        w = [rho    , rho*u         , rho*v     , rho*E                                  ] #dt
        F = [rho*u  , rho*u*u + P   , rho*u*v   , (rho*E+P)*u  -k*tf.gradients(T,x)[0]   ] #dx
        G = [rho*v  , rho*u*v       , rho*v*v +P, (rho*E+P)*v  -k*tf.gradients(T,y)[0]   ] #dy
        H = [rho*v/y, rho*u*v/y     , rho*v*v/y , (rho*E+P)*v/y                          ]
        H_v = [0, H_v2, H_v3, H_v4]        

        alpha = 0
        
        # autodiff residual 1
        e_1 = tf.gradients(F[0], x)[0] + tf.gradients(G[0], y)[0] - alpha/y*(H[0] - H_v[0])
        # autodiff residual 2
        e_2 = tf.gradients(F[1], x)[0] + tf.gradients(G[1], y)[0] - alpha/y*(H[1] - H_v[1])
        # autodiff residual 3
        e_3 = tf.gradients(F[2], x)[0] + tf.gradients(G[2], y)[0] - alpha/y*(H[2] - H_v[2])
        # autodiff residual 4
        e_4 = tf.gradients(F[3], x)[0] + tf.gradients(G[3], y)[0] - alpha/y*(H[3] - H_v[3])
        # state residual
        state_res = P - rho*R*T
        
        total_res = tf.reduce_mean(tf.square(e_1)) + \
                    tf.reduce_mean(tf.square(e_2)) + \
                    tf.reduce_mean(tf.square(e_3)) + \
                    tf.reduce_mean(tf.square(e_4)) + \
                    tf.reduce_mean(tf.square(state_res))

        return P, rho, u, v, T, total_res 
        
    def callback(self, loss):
        loss_vector.append(loss)
        print('Loss: %.3e' % (loss))
      
    def train(self, nIter): 

        tf_dict = {self.P_back_tf: self.P_back, self.x_tf: self.x, self.y_tf: self.y,
                    self.P_tf: self.P, self.rho_tf: self.rho, self.u_tf: self.u, self.v_tf: self.v,self.T_tf: self.T
                    }
        
        global loss_vector
        loss_vector = []

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            loss_value = self.sess.run(self.loss, tf_dict)

            loss_vector.append(loss_value)

            
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                # res1 = self.sess.run(self.e1, tf_dict)
                # res2 = self.sess.run(self.e2, tf_dict)
                # res3 = self.sess.run(self.e3, tf_dict)
                print('Iter: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                # print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                #     %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
                start_time = time.time()


                self.saver.save(self.sess,self.chkpt_name,global_step = it)

        
    
        self.optimizer.minimize(self.sess,
                               feed_dict = tf_dict,
                               fetches = [self.loss],
                               loss_callback = self.callback)

        return loss_vector
            
    
    def predict(self, P_back_test, x_test, y_test):

        tf_dict     = {self.P_back_tf: P_back_test, self.x_tf: x_test, self.y_tf: y_test}
        P_test      = self.sess.run(self.P_pred, tf_dict)
        rho_test    = self.sess.run(self.rho_pred, tf_dict)
        u_test      = self.sess.run(self.u_pred, tf_dict)
        v_test      = self.sess.run(self.v_pred, tf_dict)
        T_test      = self.sess.run(self.T_pred, tf_dict)

        return P_test, rho_test, u_test, v_test, T_test


    def save(self,path):
        input_dict = {"P_back": self.P_back_tf, 
                        "x": self.x_tf, 
                        "y": self.y_tf}

        output_dict = {"P": self.P_tf,
                        "rho": self.rho_tf,
                        "u":self.u_tf,
                        "v": self.v_tf,
                        "T":self.T_tf}

        tf.compat.v1.saved_model.simple_save(self.sess, path,input_dict,output_dict)



    def load(self,path):        
        tf.saved_model.load(self.sess, [tf.saved_model.tag_constants.SERVING], path)


