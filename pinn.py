import tensorflow as tf
import numpy as np
import time
import utils as utils 


class DeepPINN_2D:
    # Initialize the class
    
    def __init__(self, P_back, x, y, P, rho, u, v, Et, layers):
    
        #def __init__(self, P_back, x, y, P, rho, u, v, Et,layers):
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
        self.Et     = Et

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
        

        self.loss_vector, self.step_vector = [],[]

        self.learning_rate = tf.compat.v1.placeholder(tf.float32,shape=[])

        self.P_back_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.P_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.rho_tf     = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.v_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.Et_tf       = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        self.P_norm      = np.amax(self.P)
        self.rho_norm    = np.amax(self.rho)
        self.u_norm      = np.amax(self.u)
        self.v_norm      = np.amax(self.v)
        self.Et_norm      = np.amax(self.Et)
        self.e_norm      = 1#np.amax(P/y)
        self.e_1_norm    = 1
        self.e_2_norm    = 1
        self.e_3_norm    = 1
        self.e_4_norm    = 1

        self.P_pred, self.rho_pred, self.u_pred,self.v_pred, self.Et_pred, \
        self.e_1,self.e_2,self.e_3,self.e_4 = self.net_NS(self.P_back_tf, self.x_tf,self.y_tf)


        self.e_P    = tf.reduce_sum(tf.square((self.P_tf - self.P_pred)))
        self.e_rho  = tf.reduce_sum(tf.square((self.rho_tf - self.rho_pred)))
        self.e_u    = tf.reduce_sum(tf.square((self.u_tf - self.u_pred)))
        self.e_v    = tf.reduce_sum(tf.square((self.v_tf - self.v_pred)))
        self.e_Et   = tf.reduce_sum(tf.square((self.Et_tf - self.Et_pred)))
        self.e_1    = tf.reduce_sum(tf.square(self.e_1))
        self.e_2    = tf.reduce_sum(tf.square(self.e_2))
        self.e_3    = tf.reduce_sum(tf.square(self.e_3))
        self.e_4    = tf.reduce_sum(tf.square(self.e_4))
        #self.e_5    = tf.reduce_sum(tf.square(self.e_5))
        
        self.loss = 1*self.e_P + \
                    1*self.e_rho + \
                    1*self.e_u + \
                    1*self.e_v + \
                    1*self.e_Et + \
                    0*self.e_1 + \
                    0*self.e_2 + \
                    0*self.e_3 + \
                    0 *self.e_4 
                    #0*self.e_5/self.e_5_norm**2 


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 15000,
                                                                           'maxfun': 15000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate= 1E-3)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(save_relative_paths=True)

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
        P_rho_u_v_Et = self.neural_net(tf.concat([P_back, x, y], 1), self.weights, self.biases)
  
        P   = P_rho_u_v_Et[:,0:1]
        rho = P_rho_u_v_Et[:,1:2]
        u   = P_rho_u_v_Et[:,2:3]
        v   = P_rho_u_v_Et[:,3:4]
        Et  = P_rho_u_v_Et[:,4:5]

        ## fluid constants and relations   
        isViscous = 0
        isTransient = 0
        mu = 1.7894*10**(-5)
        R = 287
        gamma = 1.4
        k = 0.0242
        E = Et
        T = (E -  (u*u+v*v)/2)*(gamma-1)/R 

        ## AutoDiff gradients to evaluates necessary values
        sigma_xx = 2/3*mu*(tf.gradients(u, x)[0] + tf.gradients(v, y)[0]) - 2*mu*tf.gradients(u, x)[0]
        sigma_yy = 2/3*mu*(tf.gradients(u, x)[0] + tf.gradients(v, y)[0]) - 2*mu*tf.gradients(v, y)[0]
        tau_xy = - mu*(tf.gradients(u, y)[0] + tf.gradients(v, x)[0])
        tau_yx = tau_xy
        tau_thetatheta = mu*(-2/3*(tf.gradients(u, x)[0] + tf.gradients(v, y)[0]) + 4/3*v/y)
        H_v2 = - tau_xy - 2/3*y*tf.gradients(u*v/y, x)[0]
        H_v3 = - sigma_yy - tau_thetatheta - 2/3*mu*v/y - y*2/3*tf.gradients(u*v/y, y)[0]
        H_v4 = - u*tau_xy - v*sigma_yy + k*tf.gradients(T, y)[0] - 2/3*mu*v*v/y - y*tf.gradients(2/3*mu*v*v/y, y)[0] - y*tf.gradients(2/3*mu*u*v/y, x)[0]
        
        ## Conservative form terms
        w = [rho, rho*u, rho*v, rho*E]
        F = [rho*u, rho*u*u + P + isViscous*sigma_xx, rho*u*v + isViscous*tau_xy, (rho*E+P)*u + isViscous*u*sigma_xx + isViscous*v*tau_xy - isViscous*k*tf.gradients(T, x)[0]]
        G = [rho*v, rho*u*v + isViscous*tau_yx, rho*v*v + P + isViscous*sigma_yy, (rho*E+P)*v + isViscous*u*tau_yx + isViscous*v*sigma_yy - isViscous*k*tf.gradients(T, y)[0]]
        H = [rho*v/y, rho*u*v/y, rho*v*v/y, (rho*E+P)*v/y]
        H_v = [0, H_v2, H_v3, H_v4]        
        
        ## switch
        alpha = 0 ## 0 corresponds to PLANAR, 1 to AXISYMMETRIC
        
        if isTransient:
            # autodiff residual 1
            e_1 = tf.gradients(w[0], t)[0] + tf.gradients(F[0], x)[0] + tf.gradients(G[0], y)[0] - alpha/y*(H[0] - H_v[0])
            # autodiff residual 2
            e_2 = tf.gradients(w[1], t)[0] + tf.gradients(F[1], x)[0] + tf.gradients(G[1], y)[0] - alpha/y*(H[1] - H_v[1])
            # autodiff residual 3
            e_3 = tf.gradients(w[2], t)[0] + tf.gradients(F[2], x)[0] + tf.gradients(G[2], y)[0] - alpha/y*(H[2] - H_v[2])
            # autodiff residual 4
            e_4 = tf.gradients(w[3], t)[0] + tf.gradients(F[3], x)[0] + tf.gradients(G[3], y)[0] - alpha/y*(H[3] - H_v[3])
        else:
            # autodiff residual 1
            e_1 = tf.gradients(F[0], x) + tf.gradients(G[0], y)# - alpha/y*(H[0] - H_v[0])
            # autodiff residual 2
            e_2 = tf.gradients(F[1], x) + tf.gradients(G[1], y)# - alpha/y*(H[1] - H_v[1])
            # autodiff residual 3
            e_3 = tf.gradients(F[2], x) + tf.gradients(G[2], y)# - alpha/y*(H[2] - H_v[2])
            # autodiff residual 4
            e_4 = tf.gradients(F[3], x) + tf.gradients(G[3], y)# - alpha/y*(H[3] - H_v[3])
        
        # state residual
        e_5 = 0#P - rho*R*T

        return P,rho,u,v,Et, e_1, e_2,e_3,e_4 
        
    def callback(self, loss):
        self.loss_vector.append(loss)
        self.step_vector.append(1)
        print('Loss: %.3e' % (loss))

    def train(self, num_epochs, num_iter, batch_size, learning_rate): 
        switch = True
        # for epoch in range(num_epochs):
        for epoch in range(num_epochs):

            start_time = time.time()

            A = np.random.choice(range(self.x.shape[0]), size=(batch_size,), replace=False)

            for it in range(num_iter):

                #for it in range(0,N_nodes,batch_size):
                #node_idx = nodes_perm[np.arange(it,it+batch_size)]

                #slice data
                P_back_batch    = self.P_back[A].flatten()[:,None]
                x_batch = self.x[A].flatten()[:,None]
                y_batch = self.y[A].flatten()[:,None]
                P_batch = self.P[A].flatten()[:,None]
                rho_batch = self.rho[A].flatten()[:,None]
                u_batch = self.u[A].flatten()[:,None]
                v_batch = self.v[A].flatten()[:,None]
                Et_batch = self.Et[A].flatten()[:,None]


                tf_dict = {self.P_back_tf: P_back_batch, self.x_tf: x_batch, self.y_tf: y_batch,
                            self.P_tf: P_batch, self.rho_tf: rho_batch, self.u_tf: u_batch, self.v_tf: v_batch,self.Et_tf: Et_batch,
                            self.learning_rate: learning_rate}
        

                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 100 == 0:

                    elapsed = time.time() - start_time

                    loss_value = self.sess.run([self.loss],tf_dict)
                    e_1, e_2, e_3,e_4 = self.sess.run([self.e_1,self.e_2,self.e_3,self.e_4],tf_dict)
                    e_P, e_rho, e_u,e_v,e_Et = self.sess.run([self.e_P,self.e_rho,self.e_u,self.e_v,self.e_Et],tf_dict)

                    # if e_P < 0.05 and switch:
                    #     switch = False
                    #     self.loss = 1*self.e_P + \
                    #                 1*self.e_rho + \
                    #                 1*self.e_u + \
                    #                 1*self.e_v + \
                    #                 1*self.e_T + \
                    #                 0*self.e_1 + \
                    #                 0*self.e_2 + \
                    #                 0*self.e_3 + \
                    #                 0*self.e_4 

                    #     self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 

                    # res1 = self.sess.run(self.e1, tf_dict)
                    # res2 = self.sess.run(self.e2, tf_dict)
                    #res3 = self.sess.run(self.total_res, tf_dict)
                    #print(res3)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f' % 
                        (epoch, it, loss_value[0], elapsed))

                    print("\tE_1: %.3f, E_2: %.3f, E_3: %.3f, E_4: %.3f, E_5: %.3f"%
                        (e_1,e_2,e_3,e_4,0.0))

                    print("\tE_P: %.3f, E_rho: %.3f, E_u: %.3f, E_v: %.3f, E_Et: %.3f"%
                        (e_P,e_rho,e_u,e_v,e_Et))


                    # print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                    #     %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
                    
                    start_time = time.time()
                    self.saver.save(self.sess,self.ckpt_name,global_step = epoch)

                    self.loss_vector.append(loss_value[0])
                    self.step_vector.append(1)

                if epoch % 5 == 0 and it == 0:
                    path2 = self.ckpt_name + '_temp_loss.csv'
                    utils.writeLoss(path2,self.loss_vector,self.step_vector)
    
        self.optimizer.minimize(self.sess,
                            feed_dict = tf_dict,
                            fetches = [self.loss],
                            loss_callback = self.callback)


            
    def predict(self, P_back_test, x_test, y_test):

        tf_dict     = {self.P_back_tf: P_back_test, self.x_tf: x_test, self.y_tf: y_test}
        P_test      = self.sess.run(self.P_pred, tf_dict)
        rho_test    = self.sess.run(self.rho_pred, tf_dict)
        u_test      = self.sess.run(self.u_pred, tf_dict)
        v_test      = self.sess.run(self.v_pred, tf_dict)
        Et_test      = self.sess.run(self.Et_pred, tf_dict)

        return P_test, rho_test, u_test, v_test, Et_test
