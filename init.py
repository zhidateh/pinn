import os
import sys
import csv
import numpy as np

#Load all data from a directory
#Specify the path of the directory
def loadData(path):
    
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
                    for row in reader:
                        
                        #ignore header row
                        if isFirst:
                            isFirst = False
                            continue

                        #column 1: node number
                        #column 2: x-coordinate
                        #column 3: y-coordinate
                        #column 4: pressure
                        #column 5: density
                        #column 6: x-velocity, u
                        #column 7: y-velocity, v
                        #column 8: temperature
                        #column 9: total energy
                        #column 10: internal energy

                        P_back.append(float(P_b))
                        x.append(float(row[1]))
                        y.append(float(row[2]))
                        P.append(float(row[3]))
                        rho.append(float(row[4]))
                        u.append(float(row[5]))
                        v.append(float(row[6]))
                        T.append(float(row[7]))
                        Et.append(float(row[8]))
                        E.append(float(row[9]))


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

    print("Successfully loaded %d dataset(s)\n" %len(os.listdir(path)))
    return P_back,x,y,P,rho,u,v,T,E,Et


if __name__ == '__main__':
    wd = os.getcwd() + '/case/'
    loadData(wd)



