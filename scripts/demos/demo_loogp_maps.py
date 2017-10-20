import time, sys, os
sys.path.append('../../')
import matplotlib
if True: matplotlib.use('agg') # define if running code on server (True)
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


gpitch.amtgp.init_settings(visible_device=sys.argv[1], interactive=True) #  confi gpu usage, plot
data_location = '../../../datasets/maps/sample_rate_16khz/'  # location of data, params, results dir
params_location = '../../../results/files/params_activations/'
test_data_location = '../../../datasets/maps/test_data/'
results_files_location = '../../../results/files/pitch_detection/'
results_figures_location = '../../../results/figures/pitch_detection/'

pitch_detect = np.asarray(['60'])  # pitch to detect
pitch_others = np.asarray(['64', '67', '72', '76'])  # other pitches in the mixture
# fnl: filename list, td: training data, pd: pitch to detect, po: others pitches
fnl_pd, td_pd, params_pd = gpitch.amtgp.load_pitch_params_data(pitch_detect, data_loc=data_location,
                                                               params_loc=params_location)
fnl_po, td_po, params_po = gpitch.amtgp.load_pitch_params_data(pitch_others, data_loc=data_location,
                                                               params_loc=params_location)
params = {'l_act1' : params_pd[0]['l_act'],
          's_act1' : params_pd[0]['s_act'],
          'l_act2' : 1.,
          's_act2' : 1.,
          'l_com1' : params_pd[0]['l_com'],
          's_com1' : params_pd[0]['s_com'],
          'f_com1' : params_pd[0]['f_com'],
          'l_com2' : aux_list,
          's_com2' : aux_list,
          'f_com2' : aux_list}
kern_com1 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com1'], lengthscale=params['l_com1'],
                                           period=1./params['f_com1'], Nh=params['s_com1'].size)
kern_com2 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com2'], lengthscale=params['l_com2'],
                                           period=1./params['f_com2'], Nh=params['s_com2'].size)
kern_act1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act1'], variance=params['s_act1'])
kern_act2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act2'], variance=params['s_act2'])
kc, ka = [kper1, kper2], [kenv1, kenv2]
maxiter, dec, ws = 10, 10, N  # maxiter, decimation factor, window size in samples
kc, ka = [kper1, kper2], [kenv1, kenv2]
model = gpitch.loopdet.LooPDet(x=x, y=y, kern_comps=kc, kern_acts=ka, ws=ws, dec=dec, whiten=True)
model.m.likelihood.noise_var
model.optimize_windowed(disp=1, maxiter=maxiter)
model.plot_results()
plt.tight_layout()
plt.savefig('../../../results/figures/demos/demo_loogp_maps.png')





























#
