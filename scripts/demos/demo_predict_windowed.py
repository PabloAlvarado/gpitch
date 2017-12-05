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
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False)

noise_var = 1e-3
N = 16000 # number of frames
f0 = 20.
x = np.linspace(0, 1, N).reshape(-1, 1)
act = signal.hann(N).reshape(-1,1)
com = np.sin(2*np.pi*f0*x) + 0.5*np.sin(2*2*np.pi*f0*x) + 0.25*np.sin(3.1*2*np.pi*f0*x)
y = act*com  + np.sqrt(noise_var)*np.random.randn(N,1)  # generate noisy data
y = (y - y.mean()) / np.max(np.abs(y))

Nc = 3  # number of "harmoncics"
var = np.random.rand(Nc)  # variances
var_scale = 1./ (4.*np.sum(var)) #rescale (sigma)
var *= var_scale
leng = 0.75  # lengthscale
freq = f0*np.asarray(range(1, Nc+ 1))
kern_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.05, variance=10.)
kern_com = gpitch.kernels.MaternSpecMix(lengthscales=leng, variances=var, frequencies=freq,
                                        Nc=Nc)

dec = 320  # decimation factor
minibatch_size = 100
z = np.vstack((x[::dec].copy(), x[-1].copy()))
m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=kern_com, kern_act=kern_act,
                       minibatch_size=minibatch_size)  # sigmoid model


m.likelihood.variance = noise_var
m.likelihood.variance.fixed = True
m.z.fixed = True
m.kern_com.fixed = True
m.kern_act.fixed = True

st = time.time()  # Inference
logt = []
logx = []
logf = []
def logger(x):
    if (logger.i % 10) == 0:
        logx.append(x)
        logf.append(m._objective(x)[0])
        logt.append(time.time() - st)
    logger.i+=1
logger.i = 1
m.x.minibatch_size = minibatch_size
m.y.minibatch_size = minibatch_size

m.optimize(method=tf.train.AdamOptimizer(learning_rate=0.01), maxiter=100, callback=logger)



M = 1600  # number of frames per window
Nw = 10  # number of windows
xlist = [x[i*M : (i+1)*M] for i in range(Nw)]

predict_com_mean = []
predict_com_var = []

predict_act_mean = []
predict_act_var = []

for window in range(Nw):
    print('predicting window ' + str(window))
    mean_f, var_f = m.predict_com(xlist[window])
    mean_g, var_g = m.predict_act(xlist[window])

    predict_com_mean.append(mean_f)
    predict_com_var.append(var_f)
    predict_act_mean.append(mean_g)
    predict_act_var.append(var_g)

predict_com_mean = np.asarray(predict_com_mean).reshape(-1,1)
predict_com_var = np.asarray(predict_com_var).reshape(-1,1)
predict_act_mean = np.asarray(predict_act_mean).reshape(-1,1)
predict_act_var = np.asarray(predict_act_var).reshape(-1,1)

mean_f_all, var_f_all = m.predict_com(x)
mean_g_all, var_g_all = m.predict_act(x)

predict_windowed = [predict_com_mean, predict_com_var, predict_act_mean, predict_act_var]
predict_all = [mean_f_all, var_f_all, mean_g_all, var_g_all]

np.savez('../../../outfile', pw=predict_windowed, pnow=predict_all, x=x, y=y)
