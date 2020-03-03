from __future__ import division

import numpy as np

from pycog import tasktools
import matplotlib.pyplot as plt         # Alfred
from matplotlib import cm as cm         # Alfred
import seaborn as sb
import shutil
import os
import cPickle as pickle

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 1
N = 100
Nout = 20

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2

def generate_trial(rng, dt, params):
    T = 1000

#    signal_time = rng.uniform(100, T - 600)
    signal_time = rng.uniform(100, T - 800)
#    delay = 500
##    delay = 800
    delay = 200
#    delay1 = 500
    width = 20
#    width = 5 # when N=1000 & Nout=50
    magnitude = 10

#    T = 100
#    signal_time = rng.uniform(10, T - 60)
#    delay = 50
#    width = 2
#    magnitude = 4

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}

    signal_time /= dt
    delay /= dt
    width /= dt

#    print "signal_time is: ",signal_time
    # Input matrix
    X = np.zeros((len(t), Nin))
    rnd_freq = rng.uniform(10, 50) # random frequency (Alfred)

    for tt in range(len(t)):
        if tt > signal_time:
#            X[tt][0] = np.sin((tt - signal_time)*rnd_freq/delay)*np.exp(-(tt - signal_time) / delay) * magnitude + magnitude
            X[tt][0] = np.sin((tt - signal_time)*rnd_freq/delay)+np.exp(-(tt - signal_time) / delay) * magnitude
#            X[tt][0] = np.exp(-(tt - signal_time) / delay) * magnitude

    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.ones((len(t), Nout)) # Mask matrix

        for i in range(Nout):
            for tt in range(len(t)):
                Y[tt][i] = np.exp( -(tt - (signal_time + delay / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude * 3
#                Y[tt][i] = np.exp( -(tt - (signal_time + delay1 / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude

        trial['outputs'] = Y

    return trial

min_error = 0.1

n_validation = 100
#n_gradient = 1
mode         = 'continuous'


if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    rng = np.random.RandomState(1234)   # Added by Alfred

    savefile = 'examples/work/data/delay_react/delay_react.pkl'
#    savefile = 'examples/work/data/run_57000_lr1em3_1_1000_50/delay_react.pkl'
#    savefile = 'examples/work/data/run_10000_lr1em3_1_1_100_10/delay_react.pkl'
#    savefile = 'examples/work/data/run_52000_lr1em3_1_100_100/delay_react.pkl'

    rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})
    trial_args = {}


    info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    Z0 = rnn.z




    heat_map = sb.heatmap(rnn.Wout)
    plt.title('Heat map of $W_{out}$ weights matrix')
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.show()

    print "Wout is: ",rnn.Wout




    node_drop_errors = np.zeros([1,N])
    node_drop_sums = np.zeros([1,N])

    rnn_zs = np.zeros([N,Nout,len(rnn.z[0])])
    rnn_rs = np.zeros([N,N,len(rnn.r[0])])  # added for seeing all hidden nodes' values
    print "spectral radius initially is: ",np.max(abs(np.linalg.eigvals(rnn.Wrec)))

    for i in range(1):
        rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})
        trial_args = {}
        for ii in range(Nout):
            for jj in range(N):
                if(ii==jj-Nout):
                    rnn.Wout[ii,jj] = 1.0
        info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
        for j in range(Nout):
            rnn_zs[i,j,:] = rnn.z[j]/np.max(rnn.z[j])
        for j in range(N):
            rnn_rs[i,j,:] = rnn.r[j]/np.max(rnn.r[j])  # added for seeing all hidden nodes' values
    for i in range(1):
#        heat_map = sb.heatmap(rnn_zs[i,:,:])
        xticks = np.linspace(0, len(rnn.t)-1, 11, dtype=np.int)
        xticklabels = [idx*0.5/1000.0 for idx in xticks]
#        heat_map = sb.heatmap(rnn_zs[i,:,:], xticklabels=xticklabels)
        heat_map = sb.heatmap(rnn_rs[i,:,:], xticklabels=xticklabels)  # added for seeing all hidden nodes' values
        plt.title('Heat map of sequential activation of neurons')
        plt.ylabel('Output neural nodes')
        plt.xlabel('Time (s)')
        heat_map.set_xticks(xticks)
        plt.show()


    for i in range(1):
        xticks = np.linspace(0, len(rnn.t)-1, 11, dtype=np.int)
        xticklabels = [idx*0.5/1000.0 for idx in xticks]
        heat_map = sb.heatmap(rnn_zs[i,:,:], xticklabels=xticklabels)
        plt.title('Heat map of sequential activation of neurons')
        plt.ylabel('Output neural nodes')
        plt.xlabel('Time (s)')
        heat_map.set_xticks(xticks)
        plt.show()



    print "spectral radius now is: ",np.max(abs(np.linalg.eigvals(rnn.Wrec)))

    plt.plot(rnn.t/tau, rnn.u[0])
    legend = ['Input']
    for j in range(Nout):
#         plt.plot(rnn.t/tau, rnn_zs[0,j,:])
         plt.plot(rnn.t/tau, rnn_rs[0,j,:])  # added for seeing all hidden nodes' values
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()