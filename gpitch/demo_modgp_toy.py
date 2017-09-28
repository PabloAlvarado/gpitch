import numpy as np
import tensorflow as tf
import gpflow
import amtgp
import modpdet
server = True #  define if running kernel on server
import matplotlib
if server:
    matplotlib.use('agg') #  configure matplotlib to be used on server
from matplotlib import pyplot as plt


amtgp.init_settings(visible_device = '1', interactive=True) #  configure gpu usage and plotting
np.random.seed(29)
fs = 16e3  # sample frequency
N = 1600  # number of samples
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  #  time
noise_var = 1.e-3
kenv = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
kper = gpflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                     variance=np.sqrt(0.5), period=1./440)
Kenv = kenv.compute_K_symm(x)
Kper = kper.compute_K_symm(x)
f = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper).reshape(-1, 1)
f /= np.max(np.abs(f))
g = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv).reshape(-1, 1)
mean = amtgp.logistic(g)*f
y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var) #  generate data

m = modpdet.ModPDet(x=x, y=y, fs=fs, ws=N, jump=20) # pitch detection model
m.model.kern1 = kper
m.model.kern2 = kenv
m.model.likelihood.noise_var = noise_var
m.model.likelihood.noise_var.fixed = True
m.model.kern1.fixed = True # component kernel
m.model.kern2.fixed = True # activation kernel
m.model.whiten = True
m.optimize(disp=1, maxiter=500)

plt.rcParams['figure.figsize'] = (18, 18)  # set plot size
zoom_limits = [x.max()/2, x.max()/2 + 0.2*x.max()]
fig, fig_array = m.plot_results(zoom_limits)
fig_array[1, 0].plot(x, f, '.k', mew=1)
fig_array[1, 1].plot(x, f, '.k', mew=1)
fig_array[2, 0].plot(x[::7], amtgp.logistic(g[::7]), '.k', mew=0.5)
fig_array[2, 1].plot(x[::7], amtgp.logistic(g[::7]), '.k', mew=0.5)
plt.savefig('../figures/demo_modgp_toy.pdf')
plt.savefig('../figures/demo_modgp_toy.png')
