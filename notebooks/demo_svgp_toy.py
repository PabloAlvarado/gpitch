import numpy as np
import tensorflow as tf
import gpflow
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import matplotlib
server = False # define if running code on server
if server:
   matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import gpitch

gpitch.amtgp.init_settings(visible_device = '0', interactive=True) #  configure gpu usage and plotting
N = 16000 # numer of data points to load
fs, y = gpitch.amtgp.wavread('../data/60_1-down.wav', start=7000, N=N) # load two seconds of data
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
ws = N # window size (in this case the complete data)
Nr = 1 # number of restart
dsamp = 80 #  downsample rate for inducing points vector

freq = 261.63
k1 = gpflow.kernels.Matern12(input_dim=1, variance=1., lengthscales=0.1)
k2 = gpflow.kernels.Cosine(input_dim=1, variance=0.25, lengthscales=1./(2.*np.pi*freq),)
kern = k1 * k2

kern_plot = kern.compute_K(x-1., np.asarray(0).reshape(-1,1))
plt.figure()
plt.plot(x-1, kern_plot)
plt.savefig('../figures/demo_sgpr_maps_kern_plot.png')

z = x[::160].copy()
m = gpflow.svgp.SVGP(X=x, Y=y, kern=kern, likelihood=gpflow.likelihoods.Gaussian(), Z=z, whiten=True)
m.likelihood.variance = 1e-3
m.likelihood.variance.fixed = True
m.kern.fixed = True
m.optimize(disp=1, maxiter=500)

mean, var = m.predict_f(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, mean)
plt.xlim([0.25, 0.3])
plt.savefig('../figures/demo_sgpr_maps.png')

print 1./(2.*np.pi*m.kern.cosine.lengthscales.value[0])





























#
