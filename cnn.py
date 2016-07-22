import cPickle
import numpy as np
from numpy import zeros, newaxis

def loadRadio():

    radioml = cPickle.load(open("2016.04C.multisnr.pkl",'rb'))

    data = {}
    allm = []

    for k in radioml.keys():
        data[k[0]] = {}
        allm.append(k[0])

    mod = sorted(set(allm))

    for m in mod:
        dat = []
        for k in radioml.keys():
            if k[0] == m :
                for sig in range(len(radioml[k])):
                    a = np.array(radioml[k][sig][0])[:, newaxis]
                    b = np.array(radioml[k][sig][1])[:, newaxis]
                    if k[1] not in data[k[0]]:
                        data[k[0]][k[1]] = []
                    data[k[0]][k[1]].append([a,b])
    X = []
    Y = []
    x = {}
    y = {} 

    mval = {}
    count = 0

    for m in mod:
        z = np.zeros((len(mod),))
        z[count] = 1     
        mval[m] = z
        for snr in data[m]:
            dat = data[m][snr]
            for d in dat[:int(len(dat)//1.5)]:
                X.append(d)
                Y.append(z)
            for d in dat[int(len(dat)//1.5):]:
                if not snr in x:
                    x[snr] = []
                    y[snr] = []
                x[snr].append(d)
                y[snr].append(z)
        count += 1   

    return X,Y,x,y,mod
