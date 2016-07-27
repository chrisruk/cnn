#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf   
import tflearn
import cPickle
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tensor import *
from cnn import *

X,Y,x,y,mod,data = loadRadio()


print(len(X),len(Y))

s=0
for k in x:
    s+= len(x[k])

print(s)


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#X,Y =  shuffle_in_unison_inplace(np.array(X),np.array(Y))

config = tf.ConfigProto(allow_soft_placement = True)

CNN1 = True

with tf.Session(config=config) as sess:

    network = input_data(shape=[2, 128,1],name="inp")

    if CNN1:
        network = conv_2d(network, 64,[1,3], activation='relu',name="conv1") #,regularizer="L2")
        network = conv_2d(network, 16,[2,3], activation='relu',name="conv2") #,regularizer="L2")
        network = fully_connected(network, 128, activation='relu',name="fully") #,regularizer="L1")
        network = dropout(network, 0.5,name="drop1")
    else:
        print("Running CNN2")
        network = conv_2d(network, 256,[1,3], activation='relu',name="conv1") #regularizer="L1")
        network = conv_2d(network, 80,[2,3], activation='relu',name="conv2") #regularizer="L1")
        network = fully_connected(network, 256, activation='relu',name="fully")
        network = dropout(network, 1-0.6,name="drop1")


    network = fully_connected(network, len(mod), activation='softmax',name="out")

    outputs = tf.constant(mod,name="outputs1")
    strout = tf.gather(outputs,tf.argmax(network,0),name="outputs2")

    network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

    model = tflearn.DNN(network,session=sess, tensorboard_verbose=0)

    ops = tf.initialize_all_variables()
    sess.run([ops,outputs])

    model.fit(X, Y, n_epoch=100, validation_set=0.0,shuffle=True,show_metric=True, batch_size=1024, run_id='radio_cnn')

    sess.run(outputs)

    save_graph(sess,"/tmp/","saved_checkpoint","checkpoint_state","input_graph.pb","output_graph.pb","out/Softmax,outputs1,outputs2")
    
    for snr in sorted(x):
        gd = 0
        z = 0
        allv = x[snr]
        for v in allv:
            if np.argmax(model.predict ( [ v ])[0]) == np.argmax(y[snr][z]):
                gd += 1
            z = z + 1
    
        print ("SNR",snr,"ACC",gd/z)




