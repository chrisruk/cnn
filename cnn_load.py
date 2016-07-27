#!/usr/bin/python2
from __future__ import division, print_function, absolute_import


import tensorflow as tf    
import numpy as np
from tensor import *
from numpy import zeros, newaxis
import cPickle
import numpy
from cnn import *

X,Y,x,y,mod,data = loadRadio()

sess, inp, out = load_graph("/tmp/output_graph.pb","inp/X:0","out/Softmax:0")

"""
for v in sess.graph.get_operations():
    print(v.name)
"""

keep = sess.graph.get_tensor_by_name("drop1/cond/dropout/keep_prob:0")

"""
for snr in sorted(x):
    gd = 0
    z = 0
    allv = x[snr]
    for v in allv:
        pred =  sess.run ( out,feed_dict={inp:[v],keep: 1.0})[0]
        #pred =  sess.run ( out,feed_dict={inp:[v]})[0]

        if np.argmax(pred) == np.argmax(y[snr][z]):
            gd += 1
        z = z + 1
    print ("SNR",snr,"ACC",gd/z)


quit()
"""

neurons =  sess.run(sess.graph.get_tensor_by_name("outputs1:0"))



f = np.fromfile(open("/tmp/out.dat"),dtype=np.complex64)
for z in range(0,500):
    v = np.array([np.array(f[z*128:z*128+128].real)[:, newaxis],np.array(f[z*128:z*128+128].imag)[:,newaxis]])
    v2 = sess.run(out,feed_dict={inp: [v], keep: 1.0})[0]
    print (neurons[np.argmax(v2)])
quit()

for z in range(500,1000):
    f = data["WBFM"][18][z]
    v2 = sess.run(out,feed_dict={inp: [f], keep: 1.0})[0]
    print (neurons[np.argmax(v2)])

quit()



#v2 = sess.run(out,feed_dict={inp: [np.array(data['BPSK'][18][0][0],data['BPSK'][18][0][1])], keep: 1.0})[0]
#print (neurons[np.argmax(v2)])





quit()

keep = sess.graph.get_tensor_by_name("drop1/cond/dropout/keep_prob:0")

#f = numpy.fromfile('/tmp/out.dat', dtype=numpy.complex64)

for N in range(0,100):
    data = f[N*128:(N*128)+128]

    dat = [data.real[:,newaxis],data.imag[:,newaxis]]
    
    pred =  sess.run ( out,feed_dict={inp:[dat],keep: 1.0})[0]
    
    print(neurons[np.argmax(pred)])


