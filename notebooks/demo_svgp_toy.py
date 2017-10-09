import numpy as np
import tensorflow as tf
import gpflow
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft, ifftshift
import matplotlib
server = True # define if running code on server
if server:
   matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import gpitch
#plt.style.use('ggplot')


def plot(m, xpred, xdata, ydata):
    mean, var = m.predict_y(xpred)
    plt.plot(xdata, ydata, '--kx', mew=2)
    plt.plot(xpred, mean, 'C0', lw=2)
    plt.fill_between(xpred[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + \
                     2*np.sqrt(var[:,0]), color='C0', alpha=0.2)
    #plt.xlim(-0.1, 1.1)


np.random.seed()
gpitch.amtgp.init_settings(visible_device = '0', interactive=True) #  configure gpu usage and plotting
# ------------------------------------------------------------------------------
N = 32000 # numer of data points to load
fs, y = gpitch.amtgp.wavread('../data/60_1-down.wav', start=7000, N=N) # load two seconds of data
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
ws = N # window size (in this case the complete data)
Y = fft(y.reshape(-1,))
S = 2./N * np.abs(Y[0:N//2])
F = np.linspace(0, fs/2., N//2)
Nc = 10
s, l, f = gpitch.amtgp.learnparams(X=F, S=S, Nh=Nc) #  Param learning Nh=#harmonics
#sig_scale = 1./ (4.*np.sum(s)) #rescale (sigma)
#s *= sig_scale

x_full = x
N_full = N
kern = gpitch.amtgp.Matern12CosineMix(variance=s, lengthscale=l, period=1./f, Nh=Nc)
k_plot_prior = kern.compute_K(x-1, np.asarray(0).reshape(-1,1))
k_hat = ifftshift(ifft(np.abs(Y)))

K_prior = fft(k_plot_prior.reshape(-1,))
Sk_prior = 2./N * np.abs(K_prior[0:N//2])

# ------------------------------------------------------------------------------
N = 3200 # numer of data points to load
fs, y = gpitch.amtgp.wavread('../data/60_1-down.wav', start=11000, N=N) # load two seconds of data
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
ws = N # window size (in this case the complete data)
dsamp = 5 #  downsample rate for inducing points vector

z = x[::dsamp].copy()
m = gpflow.svgp.SVGP(X=x, Y=y, kern=kern, likelihood=gpflow.likelihoods.Gaussian(), Z=z, whiten=True)
m.likelihood.variance = 1e-4
m.Z.fixed = True

m.likelihood.variance.fixed = True
m.kern.fixed = True
m.optimize(disp=1, maxiter=500) # init variational params

m.likelihood.variance.fixed = True
m.kern.fixed = False
m.kern.matern12cosine_1.period.fixed=True
m.kern.matern12cosine_2.period.fixed=True
m.kern.matern12cosine_3.period.fixed=True
m.kern.matern12cosine_4.period.fixed=True
m.kern.matern12cosine_5.period.fixed=True
m.kern.matern12cosine_6.period.fixed=True
m.kern.matern12cosine_7.period.fixed=True
m.kern.matern12cosine_8.period.fixed=True
m.kern.matern12cosine_9.period.fixed=True
m.kern.matern12cosine_10.period.fixed=True
m.optimize(disp=1, maxiter=10)




plt.figure()
plot(m=m, xpred=x, xdata=x, ydata=y)
#plt.xlim([0.08, 0.10])
plt.savefig('../figures/demo_svgp_prediction.png')


k_plot_pos = m.kern.compute_K(x_full-1, np.asarray(0).reshape(-1,1))

K_pos = fft(k_plot_pos.reshape(-1,))
Sk_pos = 2./N_full * np.abs(K_pos[0:N_full//2])

plt.figure()
plt.plot(x_full, k_plot_prior/np.max(np.abs(k_plot_prior)), lw=2)
plt.plot(x_full, k_plot_pos/np.max(np.abs(k_plot_pos)), lw=2)
plt.plot(x_full, k_hat/np.max(np.abs(k_hat)), lw=2)
plt.legend(['kernel prior', 'kernel posterior', 'kernel_hat'])
plt.savefig('../figures/demo_svgp_kernels.png')

fig, axarr = plt.subplots(3, sharex=True, sharey=True)
axarr[0].plot(F, Sk_prior /np.max(Sk_prior), lw=2)
axarr[1].plot(F, Sk_pos/np.max(Sk_pos), lw=2)
axarr[2].plot(F, S/np.max(S), lw=2)
axarr[0].set_xlim([0, 4000])
axarr[0].legend(['S prior'])
axarr[1].legend(['S posterior'])
axarr[2].legend(['S data'])
plt.savefig('../figures/demo_svgp_densities.png')





























#
