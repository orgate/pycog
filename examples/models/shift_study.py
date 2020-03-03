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
Nout = 10

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
    delay = 800
#    delay1 = 500
    width = 20
#    width = 5 # when N=1000 & Nout=50
    magnitude = 4

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}

    signal_time /= dt
    delay /= dt
    width /= dt

    # Input matrix
    X = np.zeros((len(t), Nin))
    rnd_freq = rng.uniform(10, 50) # random frequency (Alfred)

    for tt in range(len(t)):
        if tt > signal_time:
#            X[tt][0] = np.sin((tt - signal_time)*rnd_freq/delay)*np.exp(-(tt - signal_time) / delay) * magnitude + magnitude
#            X[tt][0] = np.sin((tt - signal_time)*rnd_freq/delay)+np.exp(-(tt - signal_time) / delay) * magnitude
            X[tt][0] = np.exp(-(tt - signal_time) / delay) * magnitude

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
    from sympy import *
    import os

    rng = np.random.RandomState(1234)   # Added by Alfred

    savefile = 'examples/work/data/delay_react/delay_react.pkl'
#    savefile = 'examples/work/data/run_57000_lr1em3_1_1000_50/delay_react.pkl'
#    savefile = 'examples/work/data/run_10000_lr1em3_1_1_100_10/delay_react.pkl'
#    savefile = 'examples/work/data/run_52000_lr1em3_1_100_100/delay_react.pkl'

    rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})
    trial_args = {}


    info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    Z0 = rnn.z
    N = len(rnn.Wrec)


    '''heat_map = sb.heatmap(rnn.Wrec)
    plt.title('Heat map of $W_{rec}$ weights matrix')
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.show()

#    plt.hist(rnn.Wrec, bins=100)
    plt.hist(np.asarray(rnn.Wrec).reshape(-1), bins=100)
    plt.xlabel('$W_{rec}$ matrix values')
    plt.ylabel('Frequency')
    plt.title('Histogram of $W_{rec}$ matrix values')
    plt.show()'''

#    node_drop_errors = np.zeros([1,N])
#    node_drop_sums = np.zeros([1,N])

    rnn_zs = np.zeros([N,Nout,len(rnn.z[0])])
    eig_vals = np.linalg.eigvals(rnn.Wrec)
    print "spectral radius initially is: ",np.max(abs(eig_vals))
    ii = 0

    rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})     # for combined rows or cols
    trial_args = {}                                           # for combined rows or cols


    for i in range(10):
        ii = i
#        rnn.Wrec = rnn.Wrec*1.5
#        rnn.Wrec[i,:] = rnn.Wrec[i,:]
#        rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})    # for individual rows or cols
#        trial_args = {}                                          # for individual rows or cols
        col = 10
        LL = float(0.8*len(rnn.Wrec))
        for j in range(int(LL)):
            rnn.Wrec[ii,j+0*int(0.8*len(rnn.Wrec))] = rnn.Wrec[ii,j+0*int(0.8*len(rnn.Wrec))]*1.2*(1.25**(j/LL))#rng.uniform(0,0.5)   # for individual rows
#            rnn.Wrec[j,ii] = rnn.Wrec[j,ii]*1.5#rng.uniform(0,0.5)  # for individual cols
        info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)

        for j in range(Nout):
            rnn_zs[ii,j,:] = rnn.z[j]/np.max(rnn.z[j])

#    eig_vals = np.linalg.eigvals(rnn.Wrec)
#   print "spectral radius now is: ",np.max(abs(eig_vals))
#    print "rnn_zs[ii,:,:] is: ",rnn_zs[ii,:,:]


#    for i in range(1):

#        new_dir = 'seq_act_col_inh_scaled_even_1_2'
#        os.makedirs('exp2/'+new_dir)
#        results_dir = 'exp2/seq_act_col_target_scaled_even_1_5'    # for individual rows or cols
#        results_dir = 'exp2/seq_act_comb_row_target_scaled_even_11'     # for combined rows or cols
        results_dir = 'exp2/increase/seq_act_col_target_scaled_inc'    # for individual rows or cols
#        results_dir = 'exp2/seq_act_comb_row_target_scaled_even_11'     # for combined rows or cols

        if not os.path.isdir(results_dir):                         # for individual rows or cols
            os.makedirs(results_dir)                               # for individual rows or cols
        num_ticks = 11
        # the index of the position of yticks
        xticks = np.linspace(0, len(rnn.t)-1, num_ticks, dtype=np.int)
        # the content of labels of these yticks
        xticklabels = [idx*0.5/1000.0 for idx in xticks]

        fig = plt.figure()
        heat_map = sb.heatmap(rnn_zs[ii,:,:], xticklabels=xticklabels)
        plt.title('Heat map of sequential activation of neurons')
        plt.ylabel('Output neural nodes')
        plt.xlabel('Time (s)')
#        plt.xlim([0, 2])
        heat_map.set_xticks(xticks)
#        plt.xticks(np.arange(0, 2, 0.1))
#        plt.show()
        fig.savefig(results_dir+'/col_{}.png'.format(ii+1))    # for individual rows or cols
#        fig.savefig(results_dir+'.png')     # for combined rows or cols



    '''heat_map = sb.heatmap(rnn.Wrec)
    plt.title('Heat map of $W_{rec}^{mod}$ weights matrix')
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.show()

#    plt.hist(rnn.Wrec, bins=100)
    plt.hist(np.asarray(rnn.Wrec).reshape(-1), bins=100)
    plt.xlabel('$W_{rec}^{mod}$ matrix values')
    plt.ylabel('Frequency')
    plt.title('Histogram of $W_{rec}^{mod}$ matrix values')
    plt.show()'''



    '''eig_vals_modi = np.linalg.eigvals(rnn.Wrec)
    print "spectral radius now is: ",np.max(abs(eig_vals_modi))

    spectral = np.max(abs(eig_vals_modi))
    axis_values = 2*spectral*(np.arange(100)/100.0)-spectral

    plt.plot(eig_vals.real,eig_vals.imag,'b.')
    #plt.plot(eig_vals_modi.real,eig_vals_modi.imag,'r.')
    plt.plot(axis_values,axis_values*0,'k--')
    plt.plot(axis_values*0,axis_values,'k--')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    #plt.title('Eigen values of the $W_{rec}$ and $W_{rec}^{mod}$')
    plt.title('Eigen values of the $W_{rec}$')
    #plt.legend(['$\lambda_{W_{rec}}$','$\lambda_{W_{rec}^{mod}}$'])
    plt.legend(['$\lambda_{W_{rec}}$'])
    plt.show()

    plt.plot(rnn.t/tau, rnn.u[0])
    legend = ['Input']
    for j in range(Nout):
         plt.plot(rnn.t/tau, rnn_zs[ii,j,:])
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()'''
