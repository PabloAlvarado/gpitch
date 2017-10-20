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

dec = 10
aux_list = np.asarray([1., 2.]).reshape(-1,)
params = {'l_act1' : 1.,
          's_act1' : 1.,
          'l_act2' : 1.,
          's_act2' : 1.,
          'l_com1' : aux_list,
          's_com1' : aux_list,
          'f_com1' : aux_list,
          'l_com2' : aux_list,
          's_com2' : aux_list,
          'f_com2' : aux_list}

# define kernels
kern_com1 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com1'],
                                           lengthscale=params['l_com1'],
                                           period=1./params['f_com1'],
                                           Nh=params['s_com1'].size)

kern_com2 = gpitch.amtgp.Matern12CosineMix(variance=params['s_com2'],
                                           lengthscale=params['l_com2'],
                                           period=1./params['f_com2'],
                                           Nh=params['s_com2'].size)

kern_act1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act1'],
                                    variance=params['s_act1'])

kern_act2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=params['l_act2'],
                                    variance=params['s_act2'])

kc = [kper1, kper2]
ka = [kenv1, kenv2]
ws = N # winsow size in samples
m = gpitch.loopdet.LooPDet(x=x, y=y, kern_comps=kc, kern_acts=ka, ws=ws, dec=dec, whiten=True)
m.optimize_windowed(disp=1, maxiter=100)
m.plot_results()
plt.tight_layout()
plt.savefig('../../../results/figures/demos/demo_loogp_toy_new.png')































#
