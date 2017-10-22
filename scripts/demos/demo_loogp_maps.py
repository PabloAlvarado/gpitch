import time, sys, os
sys.path.append('../../')
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


visible_device = sys.argv[1]  # which gpu to use
init_model = sys.argv[2].lower() == '1'  # if true (1) initialize the gpflow model, otherwise reuse existing model

gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False) #  confi gpu usage, plot
data_location = '../../../datasets/maps/sample_rate_16khz/'  # location of data, params, results dir
params_location = '../../../results/files/params_activations/'
test_data_location = '../../../datasets/maps/test_data/'
results_files_location = '../../../results/files/pitch_detection/'
results_figures_location = '../../../results/figures/pitch_detection/'

test_data_name = test_data_location + 'test_data_5_pitches.wav'
fs, y = gpitch.amtgp.wavread(test_data_name, mono=False) # load test_data
N = y.size
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)

all_pitches = ['60', '64', '67', '72', '76']
for i in range(len(all_pitches)):
    aux = list(all_pitches)
    aux.pop(i)
    pitch_detect = np.asarray([all_pitches[i]])  # pitch to detect
    pitch_others = np.asarray(aux)  # other pitches in the mixture

    # fnl: filename list, td: training data, pd: pitch to detect, op: others pitches
    fnl_dp, td_dp, params_dp = gpitch.amtgp.load_pitch_params_data(pitch_detect, data_loc=data_location,
                                                                   params_loc=params_location)
    fnl_op, td_op, params_op = gpitch.amtgp.load_pitch_params_data(pitch_others, data_loc=data_location,
                                                                   params_loc=params_location)
    s_act_op = np.zeros(pitch_others.size)
    l_act_op = np.zeros(pitch_others.size)
    s_com_op = [None]*pitch_others.size
    l_com_op = [None]*pitch_others.size
    f_com_op = [None]*pitch_others.size

    for i in range(pitch_others.size):  # take mean for lengthscale and variance of activations
        s_act_op[i] = params_op[i]['s_act']
        l_act_op[i] = params_op[i]['l_act']
        s_com_op[i] = params_op[i]['s_com']
        l_com_op[i] = params_op[i]['l_com']
        f_com_op[i] = params_op[i]['f_com']

    params = {'l_act1' : params_dp[0]['l_act'],
              's_act1' : params_dp[0]['s_act'],
              'l_act2' : params_dp[0]['l_act'],
              's_act2' : params_dp[0]['s_act'],
              'l_com1' : params_dp[0]['l_com'],
              's_com1' : params_dp[0]['s_com'],
              'f_com1' : params_dp[0]['f_com'],
              'l_com2' : np.vstack(l_com_op),
              's_com2' : np.vstack(s_com_op),
              'f_com2' : np.vstack(f_com_op)}

    kern_com1 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com1'], lengthscale=params['l_com1'],
                                               period=1./params['f_com1'], Nh=params['s_com1'].size)
    kern_com2 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com2'], lengthscale=params['l_com2'],
                                               period=1./params['f_com2'], Nh=params['s_com2'].size)
    kern_act1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act1'], variance=params['s_act1'])
    kern_act2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act2'], variance=params['s_act2'])
    kc, ka = [kern_com1, kern_com2], [kern_act1, kern_act2]
    maxiter, dec, ws = 250, 160, N  # maxiter, decimation factor, window size in samples
    model = gpitch.loopdet.LooPDet(x=x, y=y, kern_comps=kc, kern_acts=ka, ws=ws, dec=dec, whiten=False)
    model.m.likelihood.noise_var = 1e-4
    model.m.kern_g1.fixed = False
    model.m.kern_g2.fixed = False
    model.optimize_windowed(disp=1, maxiter=maxiter)
    #model.m.kern_g2.fixed = True
    #model.m.whiten = True
    #model.optimize_windowed(disp=1, maxiter=maxiter)
    model.save_results('../../../results/files/demos/loogp/results_maps_pitch_'+ pitch_detect[0])






























#
