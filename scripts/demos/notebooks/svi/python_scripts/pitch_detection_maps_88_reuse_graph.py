import sys
import os
import time
import numpy as np
import tensorflow as tf
import gpflow
import soundfile
import pickle
sys.path.append('../../../../../')
import gpitch
from gpitch.amtgp import logistic
from gpitch import myplots
import soundfile
import peakutils
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
from scipy import signal


visible_device = sys.argv[1]  # configure gpu usage
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False)

pickleloc = '../../../../../../results/files/svi/script/'  # location saved models
filename = '../../../../../../datasets/maps/test_data/segment4-down.wav'  # loc test data

bounds = [21, 109]  # pitches to detect
midi = np.asarray([str(i) for i in range(bounds[0], bounds[1])]).reshape(-1,)  # midi list
Np = midi.size
fs = 16e3

m = [pickle.load(open(pickleloc +"maps_pitch_"+ midi[i] + ".p", "rb")) for i in range(Np)]
m_bg = pickle.load(open(pickleloc + "maps_background.p", "rb")) # load background model

y, fs = soundfile.read(filename)
y = y.reshape(-1, 1)
Ntest = y.size
x = np.linspace(0, (Ntest-1.)/fs, Ntest).reshape(-1, 1)

dec = 160  # decimation level
maxiter = 1  # max number of iterations
mbs = 200  # mini batch size
learning_rate = 0.01  # learning rate
z = x[::dec].copy()  # inducing points

all_mean_f = [None]*Np  # lists to save results prediction for the 88 notes
all_mean_g = [None]*Np
all_var_f = [None]*Np
all_var_g = [None]*Np

Nh = 20  # max number of harmonic for the component kernels
ker_com_pitch = gpitch.kernels.MaternSpecMix(Nc=Nh)  # init kernels
ker_act_pitch = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.1, variance=10.)
ker_com_bg = m_bg.kern_com
ker_act_bg = m_bg.kern_act
kc = [ker_com_pitch, ker_com_bg]
ka = [ker_act_pitch, ker_act_bg]

mloo = gpitch.loogp.LooGP(X=x, Y=y, kf=kc, kg=ka, Z=z, minibatch_size=mbs)  # init model

mloo.kern_f1.fixed = True  # fix only component kernels
mloo.kern_f2.fixed = True
mloo.kern_g1.fixed = False
mloo.kern_g2.fixed = False
mloo.likelihood.variance.fixed = False

for i in range(Np):
    print('Analysing pitch ' + str(midi[i]))

    #   update hyperparams for pitch do detect (com and act kernel)
    #   init values of variational distributions

    mloo.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)  # optimize
    mean_f, var_f, mean_g, var_g = mloo.predict_all(x)  # predict

    all_mean_f[i] = list(mean_f)  # save results on list
    all_mean_g[i] = list(mean_g)
    all_var_f[i] = list(var_f)
    all_var_g[i] = list(var_g)

    tf.reset_default_graph()  # reset graph


piano_roll = np.zeros((Np, Ntest))
for i in range(Np):
    source = logistic(all_mean_g[i][0]) * all_mean_f[i][0]
    piano_roll[i, :] = source.copy()

pickle.dump(piano_roll, open( pickleloc + "piano_roll_maps_88_more_iterations.p", "wb"))





































#
