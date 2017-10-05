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


#np.random.seed(29) # initialize randomly params
gpitch.amtgp.init_settings(visible_device = '0', interactive=True) #  configure gpu usage and plotting

Nc = 4 #  set kernel params
var =  np.ones((Nc, 1))
var = np.asarray([.5, .3, .2, .1])
var *= 1./(4.*var.sum())
beta = 0.02
f0 = 440.00
leng = np.ones((Nc, 1))
leng = np.asarray([1., 1., 1., 1.])

kern_f = gpitch.kernels.Inharmonic(input_dim=1, lengthscales=leng, variances=var, beta=beta, f0=f0)
kern_g = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=16.)

np.random.seed() # generate synthetic data
noise_var = 1e-4
N = 1600
x = np.linspace(0, 0.1, N).reshape(-1,1)
Kf = kern_f.compute_K_symm(x)
Kg = kern_g.compute_K_symm(x)
f = np.random.multivariate_normal(np.zeros(N), Kf, 1).reshape(-1, 1)
f /= np.max(np.abs(f))
g = np.random.multivariate_normal(np.zeros(N), Kg, 1).reshape(-1, 1)
mean = logistic(g) * f
y = mean + np.sqrt(noise_var)*np.random.randn(*mean.shape)

#  set model

Nc = 4
init_leng_f =1.*np.random.rand(Nc, 1)
init_var_f = 1.*np.random.rand(Nc, 1)
init_beta = np.random.rand()
init_f0 = 440.00 #  30.*np.random.rand()
init_leng_g = 1.*np.random.rand()
init_var_g = 10.*np.random.rand()
kern1 = gpitch.kernels.Inharmonic(input_dim=1, lengthscales=init_leng_f, variances=init_var_f, beta=init_beta, f0=init_f0)
kern2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.05, variance=10.)
m = gpitch.modgp.ModGP(x, y, kern_f, kern_g, x[::10].copy())
m.q_mu1.transform = gpflow.transforms.Logistic(a=-1., b=1.)
m.q_mu2.transform = gpflow.transforms.Logistic(a=-8.0, b=8.0)
m.kern1.fixed = True
m.kern2.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True
m.optimize(disp=1, maxiter=400)

m_f, v_f = m.predict_f(x)
m_g, v_g = m.predict_g(x)

nrows = 4
ncols = 2
plt.rcParams['figure.figsize'] = (24,24)  # set plot size
zoom_limits = [x.max()/2, x.max()/2 + 0.2*x.max()]

fig, fig_array = plt.subplots(nrows, ncols, sharex='row', sharey='row')

fig_array[0, 0].set_title('Data')
fig_array[0, 0].plot(x, y, lw=2)
fig_array[0, 1].set_title('Approximation')
fig_array[0, 1].plot(x, logistic(m_g)*m_f , lw=2)

fig_array[1, 0].set_title('Infered component')
fig_array[1, 0].plot(x, m_f, color='C0', lw=2)
fig_array[1, 0].fill_between(x[:, 0], m_f[:, 0] - 2*np.sqrt(v_f[:, 0]),
                     m_f[:, 0] + 2*np.sqrt(v_f[:, 0]), color='C0', alpha=0.2)

fig_array[1, 1].set_title('Actual component')
fig_array[1, 1].plot(x, f, color='C0', lw=2)

fig_array[2, 0].set_title('Infered activation')
fig_array[2, 0].plot(x, logistic(m_g), color='g', lw=2)
fig_array[2, 0].fill_between(x[:, 0], logistic(m_g[:, 0] - 2*np.sqrt(v_g[:, 0])),
                     logistic(m_g[:, 0] + 2*np.sqrt(v_g[:, 0])), color='g', alpha=0.2)

fig_array[2, 1].set_title('Actual activation ')
fig_array[2, 1].plot(x, logistic(g), color='g', lw=2)

fig_array[3, 0].set_title('Infered component (zoom in)')
fig_array[3, 0].plot(x, m_f, color='C0', lw=2)
fig_array[3, 0].fill_between(x[:, 0], m_f[:, 0] - 2*np.sqrt(v_f[:, 0]),
                     m_f[:, 0] + 2*np.sqrt(v_f[:, 0]), color='C0', alpha=0.2)
fig_array[3, 0].set_xlim(zoom_limits)

fig_array[3, 1].set_title('Actual component (zoom in)')
fig_array[3, 1].plot(x, f, color='C0', lw=2)
fig_array[3, 1].set_xlim(zoom_limits)

plt.savefig('../figures/demo_inharmonic_toy_all.png')




































#
