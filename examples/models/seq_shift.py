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
#    width = 20
    width = 20
    magnitude = 4

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}

    signal_time /= dt
    delay /= dt
    width /= dt

    X = np.zeros((len(t), Nin))

    for tt in range(len(t)):
        if tt > signal_time:
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
#mode         = 'continuous'


# Defining time shift measure for one node # Alfred
def time_shift_one_node(y_act,y_shifted,t):
    if(np.sum(y_act)<1e-2): # if actual output is almost zero everywhere # Alfred
        return (0)
    elif(np.sum(y_shifted)<1e-2): # if shifted output is almost zero everywhere # Alfred
        return (0)
    else:   # shifted output # Alfred
        return [((np.sum(y_act*t)/np.sum(y_act))-(np.sum(y_shifted*t)/np.sum(y_shifted))),(np.argmax(y_act)-np.argmax(y_shifted))*t[1]]

# Defining time shift measure for all nodes # Alfred
def time_shift_all_nodes(Y_act,Y_shifted,t):
    nodes = len(Y_act)
    time_shifts = np.zeros([2,nodes])
    for i in range(nodes):
        temp = time_shift_one_node(Y_act[i],Y_shifted[i],t)
        time_shifts[0,i] = temp[0]  # Time shift of the mean of y
        time_shifts[1,i] = temp[1]  # Time shift of the max of y
        # Nullifying large much time shifts (greater than 5*width) # Alfred
        if(abs(time_shifts[0,i])>0.5):
            time_shifts[0,i] = 0
        if(abs(time_shifts[1,i])>0.5):
            time_shifts[1,i] = 0
    return time_shifts

def rnn_run(savefile,parameters,col,value):
    rnn  = RNN(savefile, parameters)
    trial_args = {}
    for j in range(int(0.8*N)):
        rnn.Wrec[j,col] = rnn.Wrec[j,col]-value

    info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    rnn_zs = np.zeros([Nout,len(rnn.z[0])])
    for j in range(Nout):
        rnn_zs[j,:] = rnn.z[j]/np.max(rnn.z[j])
    return rnn_zs

if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    rng = np.random.RandomState(1234)   # Added by Alfred
    savefile = 'examples/work/data/delay_react/delay_react.pkl'
#    savefile = 'examples/work/data/run_10000_lr1em3_1_1_100_10/delay_react.pkl'
    parameters = {'dt': 0.5, 'var_rec': 0.01**2}

    rnn  = RNN(savefile, parameters)
    trial_args = {}
    info = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    Z0 = rnn.z

    rnn_zs0 = np.zeros([Nout,len(rnn.z[0])])
    for j in range(Nout):
        rnn_zs0[j,:] = rnn.z[j]/np.max(rnn.z[j])

    n_values = 21
    n_cols = int(0.2*N)

    shifts = np.zeros([n_cols,n_values,2])
    values = np.linspace(-0.2,0.2,n_values)
    
    for i in range(n_cols):
        print "i is: ",i
        for j in range(n_values):
            shift_temp = time_shift_all_nodes(rnn_zs0,rnn_run(savefile,parameters,int(0.8*N+i),values[j]),rnn.t/tau)
            shifts[i,j,0] = np.mean(shift_temp[0])
            shifts[i,j,1] = np.mean(shift_temp[1])
#        shifts[0,i] = np.mean(time_shift_all_nodes(rnn_zs0,rnn_run(savefile,parameters,int(0.8*N+i),0.01),rnn.t/tau))
        

    np.savetxt("shift_mean.csv", shifts[:,:,0], delimiter=",")
    np.savetxt("shift_maxima.csv", shifts[:,:,1], delimiter=",")
    

    plt.plot(shifts[:,int(n_values/5),0])
    plt.plot(shifts[:,int(2*n_values/5),0])
    plt.plot(shifts[:,int(3*n_values/5),0])
    plt.plot(shifts[:,int(4*n_values/5),0])
    plt.xlabel('Columns')
    plt.ylabel('Average shifts of means')
#    plt.legend(['-0.01','-0.05','+0.01','+0.05'])
    plt.legend([ "%.3f" % values[int(n_values/5)], "%.3f" % values[int(2*n_values/5)], "%.3f" % values[int(3*n_values/5)],"%.3f" % values[int(4*n_values/5)]])
    plt.show()


    plt.plot(shifts[:,int(n_values/5),1])
    plt.plot(shifts[:,int(2*n_values/5),1])
    plt.plot(shifts[:,int(3*n_values/5),1])
    plt.plot(shifts[:,int(4*n_values/5),1])
    plt.xlabel('Columns')
    plt.ylabel('Average shifts of maxima')
#    plt.legend(['-0.01','-0.05','+0.01','+0.05'])
    plt.legend([ "%.3f" % values[int(n_values/5)], "%.3f" % values[int(2*n_values/5)], "%.3f" % values[int(3*n_values/5)],"%.3f" % values[int(4*n_values/5)]])
    plt.show()


    plt.plot(values, shifts[int(n_cols/5),:,0].T)
    plt.plot(values, shifts[int(2*n_cols/5),:,0].T)
    plt.plot(values, shifts[int(3*n_cols/5),:,0].T)
    plt.plot(values, shifts[int(4*n_cols/5),:,0].T)
    plt.xlabel('Value added to the I/E columns')
    plt.ylabel('Average shifts of means')
#    plt.legend(['-0.01','-0.05','+0.01','+0.05'])
    plt.legend(['Column '+str(int(n_cols/5)),'Column '+str(int(2*n_cols/5)),'Column '+str(int(3*n_cols/5)),'Column '+str(int(4*n_cols/5))])
    plt.show()


    plt.plot(values, shifts[int(n_cols/5),:,1].T)
    plt.plot(values, shifts[int(2*n_cols/5),:,1].T)
    plt.plot(values, shifts[int(3*n_cols/5),:,1].T)
    plt.plot(values, shifts[int(4*n_cols/5),:,1].T)
    plt.xlabel('Value added to the I/E columns')
    plt.ylabel('Average shifts of maxima')
#    plt.legend(['-0.01','-0.05','+0.01','+0.05'])
    plt.legend(['Column '+str(int(n_cols/5)),'Column '+str(int(2*n_cols/5)),'Column '+str(int(3*n_cols/5)),'Column '+str(int(4*n_cols/5))])
    plt.show()

#            rnn_zs[i,j,:] = rnn.z[j]/1

    '''heat_map = sb.heatmap(rnn_zs[:,:])
    plt.title('Heat map of sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()

    plt.plot(rnn.t/tau, rnn.u[0])
    legend = ['Input']
    for j in range(Nout):
         plt.plot(rnn.t/tau, rnn_zs[j,:])
#         legend.append('Output {}'.format(j+1))
#        plt.plot(rnn.t/tau, rnn.r[j])
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
#    plt.legend(legend)
    plt.show()'''

