import sys
import os
import time
import numpy as np
import tensorflow as tf
import gpflow
import soundfile
import pickle
sys.path.append('../../')
import gpitch
from gpitch.amtgp import logistic
from gpitch import myplots
import soundfile
import peakutils
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
from scipy import signal


visible_device = sys.argv[1]  # configure gpu usage
pitch2detect = sys.argv[2]  # pitch to detect
pitch = int(pitch2detect)

gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False)

#pickleloc = '../../../results/files/svi/script/'  # location saved models
pickleloc = '/import/c4dm-04/alvarado/results/files/svi/script/'  # location saved models
filename = '../../../datasets/maps/test_data/MAPS_MUS-bach_846_AkPnBcht_mono.wav'  # loc test dat
bounds = [21, 109]  # pitches to detect
midi = np.asarray([str(i) for i in range(bounds[0], bounds[1])]).reshape(-1,)  # list
Np = midi.size
fs = 16e3

models = pickle.load(open(pickleloc + "maps_pitch_" + midi[pitch] + ".p", "rb"))
m_bg = pickle.load(open(pickleloc + "maps_background.p", "rb")) # load background model

y, fs = soundfile.read(filename)
y = y.reshape(-1, 1)
y /= np.max(np.abs(y))
Ntest = y.size
x = np.linspace(0, (Ntest-1.)/fs, Ntest).reshape(-1, 1)

dec = 1120  # decimation level
maxiter = 1000  # max number of iterations
mbs = 500  # mini batch size
learning_rate = 0.01  # learning rate
z = x[::dec].copy()  # inducing points

Nh = 20  # max number of harmonic for the component kernels
ker_com_pitch = gpitch.kernels.MaternSpecMix(Nc=Nh)  # init kernels
ker_act_pitch = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.1, variance=10.)
ker_com_bg = m_bg.kern_com
ker_act_bg = m_bg.kern_act
kc = [ker_com_pitch, ker_com_bg]
ka = [ker_act_pitch, ker_act_bg]

m = gpitch.loogp.LooGP(X=x, Y=y, kf=kc, kg=ka, Z=z, minibatch_size=mbs)  # init model

m.kern_f1.fixed = True  # fix only component kernels
m.kern_f2.fixed = True
m.kern_g1.fixed = False
m.kern_g2.fixed = False
m.likelihood.variance.fixed = False
m.likelihood.variance = m_bg.likelihood.variance.value.copy()  # noise learned background

#Adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

print('Se ise, detecting pitch ' + str(midi[pitch]))
dpc = models.kern_com.get_parameter_dict().copy()  # dictionary params component
dpa = models.kern_act.get_parameter_dict().copy() # dictionary params activation
m.update_params_graph(dic_par_com=dpc, dic_par_act=dpa)  # update hyperparams

m.q_mu1 = np.zeros(m.Z.shape)  # init values of variational parameters
m.q_mu2 = np.zeros(m.Z.shape)
m.q_mu3 = np.zeros(m.Z.shape)
m.q_mu4 = np.zeros(m.Z.shape)
m.q_sqrt1 = np.expand_dims(np.eye(m.Z.size), 2)
m.q_sqrt2 = np.expand_dims(np.eye(m.Z.size), 2)
m.q_sqrt3 = np.expand_dims(np.eye(m.Z.size), 2)
m.q_sqrt4 = np.expand_dims(np.eye(m.Z.size), 2)

#m.optimize(disp=0, maxiter=maxiter)  # optimize
#m.optimize(method=Adam_optimizer, maxiter=maxiter)
m.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)  # optimize

prediction = m.predict_all(x[::40].copy())  # predict

pickle.dump(prediction, open( pickleloc + "loo88/prediction_pitch_15_secs" + str(midi[pitch]) + ".p", "wb"))
