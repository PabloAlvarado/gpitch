'''This script is a demo of the new version for the modulated GP. The variable
ws defines the size (in number of samples) of the analysis window. choose ws=N
to analyze all data at once.'''
import numpy as np
from scipy import fftpack
import scipy as sp
from matplotlib import pyplot as plt
import tensorflow as tf
import GPflow
import time
import gpitch as gpi
import loogp
reload(loogp)
reload(gpi)
from sklearn.metrics import mean_squared_error as mse


plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
plt.interactive(True)
plt.close('all')

# generate synthetic data
fs = 16e3  # sample frequency
N = 2000  # number of samples
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time
noise_var = 1.e-3

pitch1 = 440.00  # Hertz, A4 (La)

kenv1 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)

kper1 = GPflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch1)

Kenv1 = kenv1.compute_K_symm(x)
Kper1 = kper1.compute_K_symm(x)

np.random.seed(29)
f1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper1).reshape(-1, 1)
f1 /= 3.*np.max(np.abs(f1))
f1 = f1 - f1.mean()
g1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv1).reshape(-1, 1)
mean = gpi.logistic(g1)*f1
y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)

s1  = fftpack.fft(mean.reshape(-1,))
T = 1. / fs
F = np.linspace(0., 0.5*fs, N/2)
S1 = 2.0/N * np.abs(s1[0:N/2])

plt.figure()
plt.plot(F, S1, '')

y1_ifft = np.fft.ifft(s1)
plt.figure()
plt.plot(mean, '.')
plt.plot(y1_ifft)

aux0 = np.fft.ifft(np.abs(s1))
aux0 = np.fft.ifftshift(aux0)
plt.figure()
plt.plot(x- x[-1]/2, aux0)


# s1  = fftpack.fft(aux0.reshape(-1,))
# T = 1. / fs
# F = np.linspace(0., 0.5*fs, N/2)
# S1 = 2.0/N * np.abs(s1[0:N/2])
#
# plt.figure()
# plt.plot(F, S1, '')

idx = np.argmax(S1)
a, b = idx - 50, idx + 50
if a < 0:
    a = 0
X, y = F[a: b,].reshape(-1,), S1[a: b,].reshape(-1,)
yhat = gpi.Lorentzian([1.0, 25., 2.*pitch1], X/(2.*np.pi))

plt.figure()
plt.plot(X, y)
plt.plot(X, yhat, 'r')

Nloop = 100
sig_v = np.linspace(0., 2., Nloop).reshape(-1,1)
lambda_v = np.linspace(0., 2., Nloop).reshape(-1,1)
objetive = np.zeros((Nloop, Nloop))

for i in range(Nloop):
    for j in range(Nloop):
        yhat = gpi.Lorentzian([sig_v[i], lambda_v[j], 2.*pitch1], X/(2.*np.pi))
        objetive[i, j] = mse(y, yhat)


plt.matshow(objetive)

























#
