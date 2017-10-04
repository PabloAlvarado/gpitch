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



gpitch.amtgp.init_settings(visible_device = '1', interactive=True) #  configure gpu usage and plotting

Nc = 4 #  set kernel params
var =  np.ones((Nc, 1))
var = np.asarray([.5, .3, .4, .1])
beta = 0.02
f0 = 20.
leng = np.ones((Nc, 1))
leng = np.asarray([.4, .3, .2, .1])

kern_f = gpitch.kernels.Inharmonic(input_dim=1, lengthscales=leng, variances=var, beta=beta, f0=f0)
kern_g = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.1, variance=10.)

np.random.seed(1) # generate synthetic data
noise_var = 1e-3
N = 2000
x = np.linspace(0, 1, N).reshape(-1,1)
Kf = kern_f.compute_K_symm(x)
Kg = kern_g.compute_K_symm(x)
f = np.random.multivariate_normal(np.zeros(N), Kf, 1).reshape(-1, 1)
g = np.random.multivariate_normal(np.zeros(N), Kg, 1).reshape(-1, 1)
mean = logistic(g) * f
y = mean + np.sqrt(noise_var)*np.random.randn(*mean.shape)

#  set model
np.random.seed() # initialize randomly params
Nc = 4
init_leng_f =1.*np.random.rand(Nc, 1)
init_var_f = 1.*np.random.rand(Nc, 1)
init_beta = np.random.rand()
init_f0 = 20.#30.*np.random.rand()
init_leng_g = 1.*np.random.rand()
init_var_g = 10.*np.random.rand()
kern1 = gpitch.kernels.Inharmonic(input_dim=1, lengthscales=init_leng_f, variances=init_var_f, beta=init_beta, f0=init_f0)
kern2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=init_leng_g, variance=init_var_g)
m = gpitch.modgp.ModGP(x, y, kern1, kern2, x[::10].copy())
m.kern1.fixed = True
m.kern2.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True
m.optimize(disp=1, maxiter=400)

mu, var = m.predict_f(x)
plt.figure()
plt.plot(x, mu, 'C0')
plt.plot(x, mu + 2*np.sqrt(var), 'C0--')
plt.plot(x, mu - 2*np.sqrt(var), 'C0--')
plt.twinx()
mu, var = m.predict_g(x)
plt.plot(x, logistic(mu), 'g')
plt.plot(x, logistic(mu + 2*np.sqrt(var)), 'g--')
plt.plot(x, logistic(mu - 2*np.sqrt(var)), 'g--')
plt.savefig('../figures/demo_inharmonic_toy_first_opt.png')


m.kern1.fixed = False
m.kern2.fixed = False
m.optimize(disp=1, maxiter=400)

mu, var = m.predict_f(x)
plt.figure()
plt.plot(x, mu, 'C0')
plt.plot(x, mu + 2*np.sqrt(var), 'C0--')
plt.plot(x, mu - 2*np.sqrt(var), 'C0--')
plt.twinx()
mu, var = m.predict_g(x)
plt.plot(x, logistic(mu), 'g')
plt.plot(x, logistic(mu + 2*np.sqrt(var)), 'g--')
plt.plot(x, logistic(mu - 2*np.sqrt(var)), 'g--')
plt.savefig('../figures/demo_inharmonic_toy_second_opt.png')


mu, var = m.predict_f(x)
plt.figure()
plt.plot(x, mu, 'C0')
plt.plot(x, mu + 2*np.sqrt(var), 'C0--')
plt.plot(x, mu - 2*np.sqrt(var), 'C0--')
plt.twinx()
mu, var = m.predict_g(x)
plt.plot(x, logistic(mu), 'g')
plt.plot(x, logistic(mu + 2*np.sqrt(var)), 'g--')
plt.plot(x, logistic(mu - 2*np.sqrt(var)), 'g--')
plt.savefig('../figures/demo_inharmonic_toy.png')

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
