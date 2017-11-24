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
import soundfile
import peakutils
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
from scipy import signal


visible_device = sys.argv[1]  # configure gpu usage
pitch2detect = sys.argv[2]  # pitch to detect
pitch = int(pitch2detect)

gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True)


m_60 = pickle.load(open("../demos/notebooks/svi/save/save_model_rwc_pitch_60.p", "rb"))
m_64 = pickle.load(open("../demos/notebooks/svi/save/save_model_rwc_pitch_64.p", "rb"))
m_67 = pickle.load(open("../demos/notebooks/svi/save/save_model_rwc_pitch_67.p", "rb"))
models = [m_60, m_64, m_67]




fs = 16000.
for i in range(3):
    x = models[i].x.value.copy()
    y = models[i].y.value.copy()
    z = models[i].z.value.copy()
    N = y.size


import tensorflow as tf
kern_com_60 = models[0].kern_com
kern_com_64 = models[1].kern_com
kern_com_67 = models[2].kern_com
kern_act_60 = models[0].kern_act
kern_act_64 = models[1].kern_act
kern_act_67 = models[2].kern_act
kern_com_60.fixed = True
kern_com_64.fixed = True
kern_com_67.fixed = True

testfile = '../../../datasets/rwc/011PFNOM_mixture.wav'
ytest, fs = soundfile.read(testfile, start=0, frames=32000)
ytest = ytest.reshape(-1,1)
ytest /= np.max(np.abs(ytest))
Ntest = ytest.size
xtest = np.linspace(0, (Ntest-1.)/fs, Ntest).reshape(-1, 1)


kern_loo = kern_com_60 + kern_com_64
kf = [kern_com_67, kern_loo]
kg = [kern_act_67, kern_act_60]
dec = 1120
maxiter = 1
minibatch_size = 500
learning_rate = 0.01
ztest = xtest[::dec].copy()
ztest.shape


# In[15]:


a, b = 0, Ntest
for i in range(1):
    if i == 0:
        kern_loo = kern_com_64 + kern_com_67
        kf = [kern_com_60, kern_loo]
        kg = [kern_act_60, kern_act_64]

    if i == 1:
        kern_loo = kern_com_60 + kern_com_67
        kf = [kern_com_64, kern_loo]
        kg = [kern_act_64, kern_act_60]

    if i == 2:
        kern_loo = kern_com_60 + kern_com_64
        kf = [kern_com_67, kern_loo]
        kg = [kern_act_67, kern_act_60]

    m = gpitch.loogp.LooGP(X=xtest, Y=ytest, kf=kf, kg=kg, Z=ztest, minibatch_size=minibatch_size)
    m.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)
    #mean_f, var_f, mean_g, var_g = m.predict_all(xtest[a:b])

    # plt.figure()
    # plt.plot(-np.array(m.logf))
    # plt.xlabel('iteration')
    # plt.ylabel('ELBO')

    # gpitch.myplots.plot_loo(mean_f=mean_f, var_f=var_f, mean_g=mean_g, var_g=var_g,
    #                         x_plot=xtest[a:b], y=ytest[a:b], z=m.Z.value, xlim=None)

    tf.reset_default_graph()
