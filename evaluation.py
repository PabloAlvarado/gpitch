import tensorflow as tf
import numpy as np
import gpitch
import os
import time
import soundfile
import matplotlib.pyplot as plt
from gpitch import window_overlap, logistic
import gpitch.myplots as mplt


def evaluation_notebook(gpu='0', inst=0, nivps=[20, 20], maxiter=[500, 500], learning_rate=[0.005, 0.001], 
                        minibatch_size=50, frames=4000, start=0, opt_za=False, window_size=4000, overlap=False, save=False):
    """
    param nivps: number of inducing variables per second, for activations and components
    """
    if save:
        print("Results are going to be saved")
    else:
        print("Results are NOT going to be saved")
    
    ## settings
    if frames < window_size:
        window_size = frames
    sess = gpitch.init_settings(gpu)  # select gpu to use
    nlinfun = gpitch.logistic_tf  # use logistic or gaussian 
    
    ## load pitch models
    directory = '/import/c4dm-04/alvarado/results/ss_amt/train/logistic/'  # location saved models
    linst = ['011PFNOM', '131EGLPM', '311CLNOM', 'ALVARADO']  # list of instruments
    instrument = linst[inst]
    pattern = instrument  # which model version
    m, names_list = gpitch.loadm(directory=directory, pattern=pattern)  
    mplt.plot_trained_models(m, instrument)

    ## load training data
    test_data_dir = '/import/c4dm-04/alvarado/datasets/ss_amt/test_data/'
    lfiles = []
    lfiles += [i for i in os.listdir(test_data_dir) if instrument + '_mixture' in i]
    xall, yall, fs = gpitch.readaudio(test_data_dir + lfiles[0], aug=False, start=start, frames=frames)
   
    if overlap:
        yall2 = np.vstack((  yall.copy(), 0.  )) 
        xall2 = np.vstack((  xall.copy(), xall[-1].copy() + xall[1].copy()  ))
        x, y = window_overlap.windowed(xall2.copy(), yall2.copy(), ws=window_size)  # return list of segments
    else:
        x, y = [xall.copy()], [yall.copy()]  
    results_list = len(x)*[None]
    var_params_list = [[], [], [], []]
    z_location_list = len(x)*[None]
    
    ## analyze whole signal by windows
    for i in range(len(y)):
        
        if i == 0: 
            ## initialize model (do this only once)
            z = gpitch.init_iv(x=x[i], num_sources=3, nivps_a=nivps[0], nivps_c=nivps[1], fs=fs)  # location inducing var
            kern = gpitch.init_kernel_with_trained_models(m)
            mpd = gpitch.pdgp.Pdgp(x[i].copy(), y[i].copy(), z, kern, minibatch_size=minibatch_size, nlinfun=nlinfun)
            mpd.za.fixed = True
            mpd.zc.fixed = True
        else:
             ## reset model to analyze a new window
            gpitch.reset_model(m=mpd, x=x[i].copy(), y=y[i].copy(), nivps=nivps, m_trained=m) 

        ## plot training data (windowed)    
        plt.figure(5), plt.title("Test data  " + lfiles[0])
        plt.plot(mpd.x.value, mpd.y.value)
        
        ## optimization
        st = time.time() 
        if minibatch_size is None:
            print ("Optimizing using VI")
            mpd.optimize(disp=True, maxiter=maxiter[0])
        else:
            print ("Optimizing using SVI")
            mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[0], epsilon=0.1), maxiter=maxiter[0])
        print("Time {} secs".format(time.time() - st))
        
        ## optimization location inducing variables
        if opt_za: 
            mpd.za.fixed = False
            st = time.time()
            if minibatch_size is None:
                print ("Optimizing location inducing variables using VI")
                mpd.optimize(disp=True, maxiter=maxiter[1])
            else:
                print ("Optimizing location inducing variables using SVI")
                mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[1], epsilon=0.1), maxiter=maxiter[1])
            print("Time {} secs".format(time.time() - st))
            mpd.za.fixed = True
    
        ## prediction
        results_list[i] = mpd.predict_act_n_com(x[i].copy())
        
        ## plot partial results
        plt.figure(123456789, figsize=(16, 4*6))
        mplt.plot_pdgp(x=x[i], y=y[i], m=mpd, list_predictions=results_list[i])
         
        ## save partial results
        var_params_list[0].append(mpd.q_mu_act)
        var_params_list[1].append(mpd.q_sqrt_act)
        var_params_list[2].append(mpd.q_mu_com)
        var_params_list[3].append(mpd.q_sqrt_com)
        z_location_list[i] = list(z)
        
        ## reset tensorflow graph
        tf.reset_default_graph()
    
    ## merge and overlap prediction results
    if overlap:
        rl_merged = window_overlap.merge_all(results_list)  # results merged
        x_final, y_final, s_final = window_overlap.get_results_arrays(x=x, y=y, sl=rl_merged[4], ws=window_size)
        final_results = [x_final, y_final, s_final]
        window_overlap.plot_sources(x_final, y_final, s_final)
    else:
        x_final = x[0].copy()
        y_final = y[0].copy()
        s_final = results_list[0][4]
        window_overlap.plot_sources(x_final, y_final, s_final)

    ##save wav files estimated sources
    if save:
        location_save = "/import/c4dm-04/alvarado/results/ss_amt/evaluation/logistic/"
        for i in range(3):
            name = names_list[i].strip('_trained.p') + "_part.wav"
            soundfile.write(location_save + name, final_results[2][i]/np.max(np.abs(final_results[2][i])), 16000)
    
    ## group results
    results = [results_list, var_params_list, z_location_list]
    return mpd, results, final_results

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
#
