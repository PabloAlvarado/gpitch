import numpy as np
import scipy as sp
from scipy import fftpack
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import GPflow
import time
import gpitch as gpi
import loogp
import sounddevice as sod #reproduce audio from numpy arrays
import soundfile  # package to load wav files
reload(loogp)
reload(gpi)


plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
plt.interactive(True)
plt.close('all')

data, fs = soundfile.read('syn_data.wav')
data = 0.5*np.sum(data, 1).reshape(-1,1)
plt.plot(data)

y_c = data[0:44100].copy()
N = np.size(y_c)

Y = fftpack.fft(y_c.reshape(-1,))
F = np.linspace(0., 0.5*fs, N/2)
S = 2.0/N * np.abs(Y[0:N/2])
plt.plot(F, S)


ker_hat =  fftpack.ifftshift(fftpack.ifft(np.abs(Y)))
plt.plot(ker_hat)


























#
