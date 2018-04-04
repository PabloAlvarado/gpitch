import tensorflow as tf
import numpy as np
import gpflow
import gpitch
import os
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
from IPython.display import display


plt.rcParams['figure.figsize'] = (16, 2)  # set plot size

def plot_results(mean_f, var_f, mean_g, var_g, x_plot, y, za, zc, xlim):
    
    mean_act = gpitch.gaussfunc(mean_g)
    plt.figure()
    plt.subplot(1, 4, 1), plt.title('data')
    plt.plot(x_plot, y)
    # plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 2), plt.title('data approx')
    plt.plot(x_plot, mean_act * mean_f, lw=2)
    # plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 3), plt.title('activation')
    plt.plot(x_plot, mean_act, 'C0', lw=2)
    plt.fill_between(x_plot, gpitch.gaussfunc(mean_g-2*np.sqrt(var_g)), gpitch.gaussfunc(mean_g+2*np.sqrt(var_g)), color='C0',
                     alpha=0.2)
    plt.plot(za, np.zeros(za.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 4), plt.title('component')
    plt.plot(x_plot, mean_f, 'C0', lw=2)
    plt.fill_between(x_plot, mean_f - 2 * np.sqrt(var_f), mean_f + 2 * np.sqrt(var_f), color='C0', alpha=0.2)
    plt.plot(zc, np.zeros(zc.shape), 'k|', mew=1)
    plt.xlim(xlim)

    
def plot_loaded_models(m, instr_name):
    for i in range(len(m)):
        x = m[i].x.value.copy()
        y = m[i].y.value.copy()
        za = m[i].za.value.copy()
        zc = m[i].zc.value.copy()
        xplot = x.reshape(-1, ).copy()
        mean_g, var_g = m[i].prediction_act
        mean_f, var_f = m[i].prediction_com
        plot_results(mean_f.reshape(-1,), var_f.reshape(-1,), mean_g.reshape(-1,), var_g.reshape(-1,), xplot, y, za, zc, 
                     xlim=[-0.01, 1.01])
        plt.suptitle(instr_name)
        

def re_init_params(m, x, y, nivps):    
    dec_a = 16000/nivps[0]
    dec_c = 16000/nivps[1]
    za1 = np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
    zc1 = np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
    za2 = 0.33*(za1[1] - za1[0]) + np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
    zc2 = 0.33*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
    za3 = 0.66*(za1[1] - za1[0]) + np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
    zc3 = 0.66*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
    
    m.Za1 = za1.copy()
    m.Za2 = za2.copy()
    m.Za3 = za3.copy()
    m.Zc1 = zc1.copy()
    m.Zc2 = zc2.copy()
    m.Zc3 = zc3.copy()
    
    m.X = x.copy()
    m.Y = y.copy()
    
    m.q_mu1 = np.zeros((zc1.shape[0], 1))  # f1
    m.q_mu2 = -np.ones((za1.shape[0], 1))  # g1
    m.q_mu3 = np.zeros((zc2.shape[0], 1))  # f2
    m.q_mu4 = -np.ones((za2.shape[0], 1))  # g2
    m.q_mu5 = np.zeros((zc3.shape[0], 1))  # f3
    m.q_mu6 = -np.ones((za3.shape[0], 1))  # g3

    q_sqrt_a1 = np.array([np.eye(za1.shape[0]) for _ in range(1)]).swapaxes(0, 2)
    q_sqrt_c1 = np.array([np.eye(zc1.shape[0]) for _ in range(1)]).swapaxes(0, 2)
    q_sqrt_a2 = np.array([np.eye(za2.shape[0]) for _ in range(1)]).swapaxes(0, 2)
    q_sqrt_c2 = np.array([np.eye(zc2.shape[0]) for _ in range(1)]).swapaxes(0, 2)
    q_sqrt_a3 = np.array([np.eye(za3.shape[0]) for _ in range(1)]).swapaxes(0, 2)
    q_sqrt_c3 = np.array([np.eye(zc3.shape[0]) for _ in range(1)]).swapaxes(0, 2)

    m.q_sqrt1 = q_sqrt_c1.copy()
    m.q_sqrt2 = q_sqrt_a1.copy()
    m.q_sqrt3 = q_sqrt_c2.copy()
    m.q_sqrt4 = q_sqrt_a2.copy()
    m.q_sqrt5 = q_sqrt_c3.copy()
    m.q_sqrt6 = q_sqrt_a3.copy()

        
def learning_on_notebook(gpu='0', inst=0, nivps=[20, 200], maxiter=[40, 10], learning_rate=[0.01, 0.001], minibatch_size=500,
                         frames=-1, start=0, opt_za=False, segmented=False, window_size=32000):
    """
    param nivps: number of inducing variables per second, for activations and components
    """
    
    if frames < window_size:
        window_size = frames
    
    
    sess = gpitch.init_settings(gpu)  # select gpu to use
     
    linst = ['011PFNOM', '131EGLPM', '311CLNOM', 'ALVARADO']  # list of instruments
    instrument = linst[inst]
    directory = '/import/c4dm-04/alvarado/results/ss_amt/train/'  # location saved models
    pattern = 'trained_25_modgp2_new_var_1' + instrument  # which model version
    
    m, names_list = gpitch.loadm(directory=directory, pattern=pattern)  # load pitch models
    plot_loaded_models(m, instrument)
    
    # load data
    test_data_dir = '/import/c4dm-04/alvarado/datasets/ss_amt/test_data/'
    lfiles = []
    lfiles += [i for i in os.listdir(test_data_dir) if instrument + '_mixture' in i]
    
    xall, yall, fs = gpitch.readaudio(test_data_dir + lfiles[0], aug=False, start=start, frames=frames)
    
    # if segmented:
    x, y = gpitch.segment(xall.copy(), yall.copy(), window_size=window_size)  # return list of segments
    # else:
        # x, y = [x], [y]  # convert to list

        
    nlinfun = gpitch.gaussfunc_tf  # use gaussian as non-linear transform for activations 
    mpd = gpitch.ssgp.init_model(x=x[0].copy(), y=y[0].copy(), m1=m[0], m2=m[1], m3=m[2], niv_a=nivps[0], niv_c=nivps[1], 
                                 minibatch_size=minibatch_size, nlinfun=nlinfun, quad=False)  # init pitch detection model
    for i in range(len(y)):
        plt.figure(5), plt.title("Test data  " + lfiles[0])
        plt.plot(x[i], y[i])
        
        if i is not 0:
            re_init_params(m=mpd, x=x[i].copy(), y=y[i].copy(), nivps=nivps)

        
        st = time.time()  # run optimization

        if minibatch_size is None:  
            print ("Using VI")
            mpd.optimize(disp=True, maxiter=maxiter[0])
        else:
            print ("Using SVI")
            mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[0], epsilon=0.1), maxiter=maxiter[0])

        print("Time optimizing {} secs".format(time.time() - st))

        mf, vf, mg, vg, x_plot, y_plot =  gpitch.ssgp.predict_windowed(x=x[i], y=y[i], predfunc=mpd.predictall)  # predict
        gpitch.myplots.plot_ssgp_gauss(mpd, mean_f=mf, var_f=vf, mean_g=mg, var_g=vg, x_plot=x_plot, y=y_plot)  # plot results

        if opt_za:
            mpd.Za1.fixed = False
            mpd.Za2.fixed = False
            mpd.Za3.fixed = False

            st = time.time()

            if minibatch_size is None:  
                print ("Using VI")
                mpd.optimize(disp=True, maxiter=maxiter[1])
            else:
                print ("Using SVI")
                mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[1], epsilon=0.1), maxiter=maxiter[1])

            print("Time optimizing location inducing variables {} secs".format(time.time() - st))

            mf, vf, mg, vg, x_plot, y_plot =  gpitch.ssgp.predict_windowed(x=x[i], y=y[i], predfunc=mpd.predictall)  # predict
            gpitch.myplots.plot_ssgp_gauss(mpd, mean_f=mf, var_f=vf, mean_g=mg, var_g=vg, x_plot=x_plot, y=y_plot)

#         print("Likelihood")
#         display(mpd.likelihood)

#         print("Activation kernels")
#         display(mpd.kern_g1)
#         display(mpd.kern_g2)
#         display(mpd.kern_g3)

#         print("Component kernels")
#         data_com_kern = pd.DataFrame({'Lengthscales':[mpd.kern_f1.lengthscales.value[0].copy(), 
#                                                       mpd.kern_f2.lengthscales.value[0].copy(), 
#                                                       mpd.kern_f3.lengthscales.value[0].copy()]})
#         display(data_com_kern)
        
    return mpd
    
    # group results
    
    # return prediction mf ,vf, mg, vg, lists with tree arrays each.
    # return vartiational parameters and location inducing variables
    #return np.pi























#