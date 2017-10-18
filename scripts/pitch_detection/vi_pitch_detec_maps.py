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
import os
sys.path.append('../../')
import gpitch
reload(gpitch)
from gpitch.amtgp import logistic

plt.figure()
for i in range(Np):
    Sk = gpitch.amtgp.LorM(x=params[i]['F'], s=params[i]['s_com'],
                           l=1./params[i]['l_com'], f=2*np.pi*params[i]['f_com'])

    plt.subplot(nrows, 2, 2*i + 1)
    plt.plot(x, train_data[i])

    plt.subplot(nrows, 2, 2*i + 2)
    plt.plot(params[i]['F'], params[i]['S'], lw=2)
    plt.twinx()
    plt.plot(params[i]['F'], Sk, '--C1', lw=2)
    plt.xlim([0, 4000])
    plt.tight_layout()
#plt.savefig(res_fig_location + 'data.png')

plt.figure(figsize=(18, 6))
plt.plot(x, y_test)
plt.tight_layout()
plt.ylim([-1.1, 1.1])
plt.xlim([0, 2])
