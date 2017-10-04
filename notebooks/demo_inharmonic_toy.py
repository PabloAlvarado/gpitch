import numpy as np
import tensorflow as tf
import gpflow
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import matplotlib
server = True # define if running code on server
if server:
   matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import gpitch
from gpitch.amtgp import logistic
reload(gpitch)


np.random.seed(1)
gpitch.amtgp.init_settings(visible_device = '0', interactive=True) #  configure gpu usage and plotting

Nc = 4 #  set kernel params
var =  np.ones((Nc, 1))
var = np.asarray([.5, .3, .4, .1])
beta = 0.02
f0 = 20.
leng = np.ones((Nc, 1))
leng = np.asarray([.4, .3, .2, .1])

kern_f = gpitch.kernels.Inharmonic(input_dim=1, lengthscales=leng, variances=var, beta=beta, f0=f0)
kern_g = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.1, variance=10.)

noise_var = 1e-3
N = 2000 # generate synthetic data
x = np.linspace(0, 1, N).reshape(-1,1)
Kf = kern_f.compute_K_symm(x)
Kg = kern_g.compute_K_symm(x)
f = np.random.multivariate_normal(np.zeros(N), Kf, 1).reshape(-1, 1)
g = np.random.multivariate_normal(np.zeros(N), Kg, 1).reshape(-1, 1)
mean = logistic(g) * f
y = mean + np.sqrt(noise_var)*np.random.randn(*mean.shape)


m = gpitch.modgp.ModGP(x, y, kern_f, kern_g, x[::10].copy())
m.kern1.fixed = True
m.kern2.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True
m.optimize(disp=1, maxiter=100)

m.kern1.fixed = False
m.kern2.fixed = False
m.optimize(disp=1, maxiter=100)


mu, var = m.predict_f(x)
plt.plot(x, mu, 'C0')
plt.plot(x, mu + 2*np.sqrt(var), 'C0--')
plt.plot(x, mu - 2*np.sqrt(var), 'C0--')
plt.twinx()
mu, var = m.predict_g(x)
plt.plot(x, logistic(mu), 'g')
plt.plot(x, logistic(mu + 2*np.sqrt(var)), 'g--')
plt.plot(x, logistic(mu - 2*np.sqrt(var)), 'g--')

plt.figure()
plt.plot(x, f, lw=2)
plt.twinx()
plt.plot(x, logistic(g), 'g', lw=2)

plt.figure()
plt.plot(x, y, lw=2)





x_plot = np.linspace(-1, 1, N).reshape(-1,1)
kern_plot = kern_f.compute_K(x_plot, np.asarray(0.).reshape(-1,1)) #  plot kernel


plt.figure()
plt.plot(x_plot, kern_plot, lw=2)
plt.xlim([-1, 1])
plt.tight_layout()
plt.savefig('../figures/demo_inharmonic_toy.png')
#print kern.compute_Kdiag(x)































#