import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
#Load all data from a directory
#Specify the path of the directory
def loadData(path,filter):
    
    #path = "%s/%s/"%(os.getcwd(),purpose)
    #path = "/home/svu/e0072438/PINN/%s"%purpose


    P_back, P, x, y, rho, u, v, T, Et, E = [],[],[],[],[],[],[],[],[],[]
    cnt = 0
    print("\nLoading mother dataset in %s..."%path)

    #check all files in the directory
    for f in os.listdir(path):
        #get correct file extension
        if f.endswith(".csv"):
            
            try:
                #Obtain magnitude of back pressure in the file name
                P_b = float(f[f.index('=')+1:-4])
                cnt += 1
                print("\t Progress: %d/%d"%(cnt, len(os.listdir(path))))

                f = path + f

                with open(f, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    isFirst = True
                    nrow = 0
                    ncol = 0

                    for row in reader:
                        #ignore header row
                        if isFirst:
                            isFirst = False
                            continue

                        if abs(float(row[1])) < 0.001:
                            nrow += 1

                        if (nrow%filter ==0):
                            #column 1: node number
                            #column 2: x-coordinate
                            #column 3: y-coordinate
                            #column 4: pressure
                            #column 5: density
                            #column 6: x-velocity, u
                            #column 7: y-velocity, v
                            #column 8: temperature
                            #column 9: total temperature
                            #column 10: total energy
                            #column 11: internal energy

                            if (ncol % filter ==0 and abs(float(row[2]))>0.00001):
                                P_back.append(float(P_b))
                                x.append(float(row[1]))
                                y.append(float(row[2]))
                                P.append(float(row[3]))
                                rho.append(float(row[4]))
                                u.append(float(row[5]))
                                v.append(float(row[6]))
                                T.append(float(row[7]))
                                Et.append(float(row[9]))
                                E.append(float(row[10]))
                        
                        ncol+=1
                            
                        #nrow +=1


            except:
                print("\t !! Warning %s contains invalid filename"%f)
                pass

    P_back  = np.asarray(P_back)
    x       = np.asarray(x)
    y       = np.asarray(y)
    P       = np.asarray(P)
    rho     = np.asarray(rho)
    u       = np.asarray(u)
    v       = np.asarray(v)
    T       = np.asarray(T)
    Et      = np.asarray(Et)
    E       = np.asarray(E)

    print("Successfully loaded %d dataset(s) in %s\n" %(len(os.listdir(path)),path))
    return P_back,x,y,P,rho,u,v,T,E,Et

def writeData(path,x,y,P,rho,u,v,T):
    
    #column 1: node number
    #column 2: x-coordinate
    #column 3: y-coordinate
    #column 4: pressure
    #column 5: density
    #column 6: x-velocity, u
    #column 7: y-velocity, v
    #column 8: temperature
    #column 9: total temperature
    #column 10: total energy
    #column 11: internal energy
    e = np.zeros(x.shape[0]).flatten()[:,None]
    data = np.concatenate((e,x,y,P,rho,u,v,T,e,e,e),axis = 1)

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        #write header
        writer.writerow(['Node number',
                        'x-coordinate',
                        'y-coordinate',
                        'pressure',
                        'density',
                        'x-velocity',
                        'y-velocity',
                        'temperature',
                        'total temperature',
                        'total energy',
                        'internal energy',])


        #write data
        writer.writerows(data)

    print("Successfully saved predicted data in %s\n" %(path))

def writeLoss(path,loss_vector,ts_vector):

    loss_vector = np.array(loss_vector).flatten()[:,None]
    ts_vector   = np.array(ts_vector).cumsum().flatten()[:,None]
    
    data = np.concatenate((ts_vector,loss_vector),axis = 1)

    with open(path,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['timestamp','loss'])
        writer.writerows(data)


def plotLoss(path,x_lower_lim,x_upper_lim,y_lower_lim,y_upper_lim):
    x,y = [],[]
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        isFirst = True
        for row in reader:
            #ignore header row
            if isFirst:
                isFirst = False
                continue

            # if float(row[1]) > 10**4:
            #     continue 

            x.append(float(row[0]))
            y.append(float(row[1]))

    print(x)
    print(y)
    plt.plot(x,y)
    plt.xlim((x_lower_lim,x_upper_lim))
    plt.ylim((y_lower_lim,y_upper_lim))
    plt.show()

def plotAll(path,x,y,P,rho,u,v,T,P_pred,rho_pred,u_pred,v_pred,T_pred):
    
    plot_x,plot_y,plot_z = [],[],[]

    for i in range(len(y)):
        if abs(y[i] - min(y)) < 0.005:
            plot_x.append(x[i])
            plot_y.append(P[i])
            plot_z.append(P_pred[i])
        else:
            break



    plot_x = np.array(plot_x).flatten().tolist()
    plot_y = np.array(plot_y).flatten().tolist()
    plot_z = np.array(plot_z).flatten().tolist()




    plt.plot(plot_x,plot_y, '*r')
    plt.plot(plot_x,plot_z, 'ob')
    plt.xlim((0,1))
    plt.ylim((0,90000))
    plt.show()


if __name__ == '__main__':
    loadData('train')



