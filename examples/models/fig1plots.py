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




    '''heat_map = sb.heatmap(Y)
    plt.title('Heat map of target sequential activation of neurons')
    plt.ylabel('Target output neural nodes')
    plt.xlabel('Time')
    plt.show()


    plt.plot(t/tau, X[0])
#    plt.plot(rnn.t/tau, rnn.r[3])    
#    plt.plot(rnn.t/tau, rnn.r[4])    
#    for j in range(Nout):
    legend = ['Input']
    for j in range(Nout):
         plt.plot(t/tau, Y[i])
         legend.append('Output {}'.format(j+1))
#        plt.plot(rnn.t/tau, rnn.r[j])
    plt.title('Target sequential activation of neurons')
    plt.ylabel('Target output neural nodes')
    plt.xlabel('Time')
    plt.legend(legend)
    plt.show()'''




    return trial

min_error = 0.1

n_validation = 100
#n_gradient = 1
#mode         = 'continuous'


if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    rng = np.random.RandomState(1234)   # Added by Alfred

#    savefile = 'examples/work/data/delay_react/delay_react.pkl'
#    savefile = 'examples/work/data/run_57000_lr1em3_1_1000_50/delay_react.pkl'
    savefile = 'examples/work/data/run_10000_lr1em3_1_1_100_10/delay_react.pkl'
#    savefile = 'examples/work/data/run_52000_lr1em3_1_100_100/delay_react.pkl'

    rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})
    trial_args = {}


    info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    Z0 = rnn.z

#    signal_time
#    delay = 500
#    width = 20
#    magnitude = 4
#    Y = np.zeros((len(t), Nout)) # Output matrix

#    for i in range(Nout):
#        for tt in range(len(t)):
#            Y[tt][i] = np.exp( -(tt - (signal_time + delay / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude


    print "rnn.Wrec is: ",rnn.Wrec
    print "sums0 are: ",np.sum(rnn.Wrec,axis=0)
    print "sums1 are: ",np.sum(rnn.Wrec,axis=1)
#    heat_map = sb.heatmap(rnn.z)
#    plt.show()

    heat_map = sb.heatmap(rnn.Wrec)
    plt.title('Heat map of $W_{rec}$ weights matrix')
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.show()

#    plt.hist(rnn.Wrec, bins=100)
    plt.hist(np.asarray(rnn.Wrec).reshape(-1), bins=100)
    plt.xlabel('$W_{rec}$ matrix values')
    plt.ylabel('Frequency')
    plt.title('Histogram of $W_{rec}$ matrix values')
    plt.show()

    print "rnn.Win is: ",rnn.Win
    print "rnn.brec is: ",rnn.brec
    print "rnn.bout is: ",rnn.bout
    print "rnn.x0 is: ",rnn.x0

    node_drop_errors = np.zeros([1,N])
    node_drop_sums = np.zeros([1,N])













    T = 1000

#    signal_time = rng.uniform(100, T - 600)
    signal_time = rng.uniform(100, T - 800)
#    delay = 500
    delay = 800
#    delay1 = 500
    width = 20
#    width = 5 # when N=1000 & Nout=50
    magnitude = 4
    dt=0.5
    var_rec=0.01**2
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

    for tt in range(len(t)):
        if tt > signal_time:
            X[tt][0] = np.exp(-(tt - signal_time) / delay) * magnitude

#    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

#    if params.get('target_output', False):
    Y = np.zeros((len(t), Nout)) # Output matrix
    M = np.ones((len(t), Nout)) # Mask matrix

    for i in range(Nout):
        for tt in range(len(t)):
            Y[tt][i] = np.exp( -(tt - (signal_time + delay / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude * 3
#                Y[tt][i] = np.exp( -(tt - (signal_time + delay1 / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude

#    trial['outputs'] = Y

    rnn_zs1 = np.zeros([N,Nout,len(rnn.z[0])])

    for j in range(Nout):
        rnn_zs1[0,j,:] = rnn.z[j]/np.max(rnn.z[j])














    colors = ['r','g','b','c','m','y','k','w','r','g']


    heat_map = sb.heatmap(rnn_zs1[0,:,:])
    plt.title('Heat map of sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()

#    print "rnn.z is: ",rnn.z
#    print "rnn.r is: ",rnn.r

    plt.plot(rnn.t[1:]/tau, rnn.u[0,1:])
#    plt.plot(rnn.t/tau, rnn.r[3])    
#    plt.plot(rnn.t/tau, rnn.r[4])    
#    for j in range(Nout):
    legend = ['Input']
    for j in range(Nout):
         plt.plot(rnn.t[1:]/tau, rnn_zs1[0,j,1:],colors[j])
         plt.plot(rnn.t[1:]/tau, Y.T[j]/max(Y.T[j]),colors[j]+'--')
#         legend.append('Output {}'.format(j+1))
#        plt.plot(rnn.t/tau, rnn.r[j])
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.legend(['Input','Target','Trial'])
    plt.show()



    rnn_zs2 = np.zeros([N,Nout,len(rnn.z[0])])

    for i in range(1):
        rnn  = RNN(savefile, {'dt': 0.5, 'var_rec': 0.01**2})
        trial_args = {}
#        rnn.Wrec = rnn.Wrec*0.5
#        rnn.Wrec[i,:] = rnn.Wrec[i,:]*0
#        print "rnn.Wrec is: ",rnn.Wrec
#        col = 6#12#16 for case 2 (100,10)
        col = 6
        for j in range(int(0.8*N)):
            rnn.Wrec[j,int(0.8*N+col)] = rnn.Wrec[j,int(0.8*N+col)]-0.04#rng.uniform(0,0.5)
#            rnn.Wrec[j,j] = rnn.Wrec[j,j]#+0.3#rng.uniform(0,0.5)
#        rnn.Wrec[8,9] = rnn.Wrec[8,9]*1.2
#        rnn.Wrec[6,6] = rnn.Wrec[6,6]+0.5
#        rnn.Wrec[9,1] += 0.5
#        rnn.Wrec[2,2] = rnn.Wrec[2,2]+0.5
#        rnn.Wrec[1,8] = rnn.Wrec[1,8]*0.05
#        rnn.Wrec[1,9] = rnn.Wrec[1,9]*2
#        rnn.Wrec[2,8] = rnn.Wrec[2,8]*1
#        rnn.Wrec[2,9] = rnn.Wrec[2,9]*0
#        rnn.Wrec[2,i+8] = rnn.Wrec[2,i+8]*1.4
#        print "rnn.Wrec for node %d is: " % (i+1),rnn.Wrec
        info1 = rnn.run(inputs=(generate_trial, trial_args), seed=200)
#        rnn_zs[i,:,:] = rnn.z
        for j in range(Nout):
            rnn_zs2[i,j,:] = rnn.z[j]/np.max(rnn.z[j])
#            rnn_zs[i,j,:] = rnn.z[j]/1
#        node_drop_errors[0,i] = np.sum((rnn_zs[i,:,:]-Z0))
#        node_drop_sums[0,i] = np.sum(rnn_zs[i,:,:])
        #print "Squared error after removing recurrent node %d is: " % (i+1),np.sum((rnn.z-Z0)**2)

#    print "node_drop_errors is: ",node_drop_errors
#    print "node_drop_sums is: ",node_drop_sums
##    print "rnn.Wrec is: ",rnn.Wrec

    heat_map = sb.heatmap(rnn_zs2[0,:,:])
    plt.title('Heat map of sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()

#    print "rnn.z is: ",rnn.z
#    print "rnn.r is: ",rnn.r

    plt.plot(rnn.t/tau, rnn.u[0])
#    plt.plot(rnn.t/tau, rnn.r[3])    
#    plt.plot(rnn.t/tau, rnn.r[4])    
#    for j in range(Nout):
    legend = ['Input']
    for j in range(Nout):
         plt.plot(rnn.t/tau, rnn_zs1[0,j,:],colors[j])
         plt.plot(rnn.t/tau, rnn_zs2[0,j,:],colors[j]+'--')
#         legend.append('Output {}'.format(j+1))
#        plt.plot(rnn.t/tau, rnn.r[j])
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.legend(['Input','Trial output','$W_{rec}$ modified'])
#    plt.legend(legend)
    plt.show()







#    heat_map = sb.heatmap(rnn_zs[5,:,:])
#    plt.show()


#    heat_map = sb.heatmap(rnn_zs[80,:,:])
#    plt.show()

#    heat_map = sb.heatmap(rnn_zs[85,:,:])
#    plt.show()



    '''plt.plot(node_drop_errors[0,:])
    plt.xlabel('Recurrent node')
    plt.ylabel('Error')
    plt.title('Error for every recurrent node dropout')
    plt.show()

    plt.plot(node_drop_sums[0,:])
    plt.xlabel('Recurrent node')
    plt.ylabel('Normalized error')
    plt.title('Normalized error for every recurrent node dropout')
    plt.show()'''


#    for i in range(N):
#        heat_map = sb.heatmap(rnn_zs[i,:,:])
#        plt.show()
#        print "rnn.Wrec[%d]" % (i+1),rnn.Wrec[i,:]


##    info = rnn.run(inputs=(generate_trial, trial_args), seed=200)

#    print "rnn.t/tau is: ",rnn.t/tau                # Alfred
#    print "rnn.u[0] is: ",rnn.z[0]                  # Alfred
#    print len(rnn.z)




    '''if savefile is not None:
        # Check that file exists
        if not os.path.isfile(savefile):
            print("[ {}.RNN ] File {} doesn't exist.".format(THIS, savefile))
            sys.exit(1)

        # Ensure we have a readable file
        base, ext = os.path.splitext(savefile)
        savefile_copy = base + '_copy' + ext
        while True:
            shutil.copyfile(savefile, savefile_copy)
            try:
                with file(savefile_copy, 'rb') as f:
                    save = pickle.load(f)
                break
            except EOFError:
                wait = 5
                print("[ {}.RNN ] Got an EOFError, trying again in {} seconds."
                      .format(THIS, wait))
                time.sleep(wait)


    #print "save['params'] is: ",save['params']
    print "save['best']['params'][1] is: ",save['best']['params'][1]
    Win, Wrec, Wout, brec, bout, x0 = save['best']['params']'''

##    Win, Wrec, Wout, brec, bout, x0 = rnn.Win, rnn.Wrec, rnn.Wout, rnn.brec, rnn.bout, rnn.x0


##    rnnz_norm = rnn.z
#    print "Win is: ",Win
#    print "Wrec is: ",Wrec
#    print "Wout is: ", Wout
#    print "brec is: ",brec
#    print "bout is: ",bout
#    print "x0 is: ",x0

##    for i in range(Nout):
##        print "Wout[i,:] is: ",Wout[i,:]

##    for i in range(Nout):
##        rnnz_norm[i] = rnn.z[i]/np.max(rnn.z[i])

    '''heat_map = sb.heatmap(rnnz_norm)
    plt.title('Heatmap of sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()

    plt.plot(rnn.t/tau, rnn.u[0])
    for i in range(Nout):
         plt.plot(rnn.t/tau, rnnz_norm[i])
    plt.title('Sequential activation of neurons')
    plt.ylabel('Output neural nodes')
    plt.xlabel('Time')
    plt.show()


    Wrec_list = []
    log_Wrec_list = []

    for i in range(N):
        for j in range(N):
            if(Wrec[i,j]>0):
                Wrec_list.append(Wrec[i,j])
                log_Wrec_list.append(np.log(Wrec[i,j]))

    plt.hist(log_Wrec_list,bins=100)
    plt.xlabel('$log(W_{rec})$ matrix values')
    plt.ylabel('Frequency')
    plt.title('Histogram of $log(W_{rec})$ matrix values')
    plt.show()'''
    '''fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.u[0], color=Figure.colors('blue'))
    for i in range(Nout):
        plot.plot(rnn.t/tau, rnn.z[i], color=Figure.colors('red'))
    plot.xlim(rnn.t[0]/tau, rnn.t[-1]/tau)
    plot.ylim(0, 15)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel(r'$t/\tau$')

    fig.save(path='.', name='delay_react')'''
