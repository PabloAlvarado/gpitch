import sys
import os
import pickle
import numpy as np
import tensorflow as tf
import gpflow
sys.path.append('../../../../../')
import gpitch
from gpitch.amtgp import logistic
from gpitch import myplots
import peakutils
from scipy.fftpack import fft


visible_device = sys.argv[1]  # configure gpu usage
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False)

saveloc = '../../../../../../results/files/svi/script/'
fname = '../../../../../../datasets/maps/test_data/segment4-down.wav'
dec = 160  # decimation factor
minibatch_size = 200  # batch size svi
maxiter = 5000  #  maximun number of iterations in optimization
learning_rate = 0.01  # learning rate svi optimization

y, fs = gpitch.amtgp.wavread(filename=fname, mono=False)  #  the signal is already mono
N = y.size
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time vector

Y = fft(y.reshape(-1,))
S = 2./N * np.abs(Y[:N/2])
F = np.linspace(0, fs/2., N/2)
S /= S.sum()

idx = peakutils.indexes(S, thres=0.0005/max(S), min_dist=100)
F_star, S_star = F[idx], S[idx]
var_scale = S_star.sum()
S_star /= 4.*var_scale
Nc = F_star.size

z = np.vstack((x[::dec].copy(), x[-1].copy()))

kern_com = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=0.1, variances=S_star,
                                        frequencies=F_star, Nc=Nc)
kern_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.25, variance=10.)

m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=kern_com, kern_act=kern_act, whiten=True,
                       minibatch_size=minibatch_size)

m.fixed_msmkern_params(freq=False, var=True)
m.kern_com.lengthscales.fixed = False
m.kern_com.lengthscales.transform = gpflow.transforms.Logistic(0., 10.0)
m.kern_act.fixed = False
m.likelihood.variance.fixed = False
m.z.fixed = True

m.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)
pickle.dump(m, open(saveloc + "maps_background" + ".p", "wb"))
