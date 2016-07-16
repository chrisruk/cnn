#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import
from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2
import threading
import time
import numpy
import struct
import numpy as np
import tensorflow as tf   
import specest 
from tensor import *
import matplotlib.pylab as plt
import tflearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cPickle
import time
from tensor import *
from numpy import zeros, newaxis
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


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
                a = numpy.array(radioml[k][sig][0])[:, newaxis]
                b = numpy.array(radioml[k][sig][1])[:, newaxis]
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
        for d in dat[:len(dat)//2]:
            X.append(d)
            Y.append(z)
        for d in dat[len(dat)//2:]:
            if not snr in x:
                x[snr] = []
                y[snr] = []
            x[snr].append(d)
            y[snr].append(z)
    count += 1    

network = input_data(shape=[None, 2, 128,1])
network = conv_2d(network, 64,[1,3], activation='relu')
network = conv_2d(network, 16,[2,3], activation='relu')
network = fully_connected(network, 128, activation='relu')
network = fully_connected(network, len(mod), activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=100, shuffle=True,show_metric=True, batch_size=1024)

for snr in sorted(x):
    gd = 0
    z = 0
    allv = x[snr]
    for v in allv:
        if np.argmax(model.predict ( [ v ])[0]) == np.argmax(y[snr][z]):
            gd += 1
        z = z + 1
    
    print ("SNR",snr,"ACC",gd/z)




