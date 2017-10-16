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
sys.path.append('../../../')
import gpitch
import os

visible_device = sys.argv[1] #  load external variable (gpu to use)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #  deactivate tf warnings
gpitch.amtgp.init_settings(visible_device = visible_device, interactive=True) #  configure gpu usage and plotting


location = "../../../../datasets/maps/sample_rate_16khz/" # load list of files no analyse
lfiles = gpitch.amtgp.load_filename_list(location + 'filename_list.txt')

for nfile in range(lfiles.size):
    N = 32000 # numer of data points to load
    fs, y = gpitch.amtgp.wavread(location + lfiles[nfile] + '.wav', start=5000, N=N) # load two seconds of data
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
    Y = fft(y.reshape(-1,)) #  FFT data
    S =  2./N * np.abs( Y[0:N//2] ) #  spectral density data
    F = np.linspace(0, fs/2., N//2) #  frequency vector
    Nc = 10 #  maximun number of frequency components to select
    s, l, f = gpitch.amtgp.learnparams(X=F, S=S, Nh=Nc) #  param learning
    Sker = gpitch.amtgp.LorM(x=F, s=s, l=1./l, f=2*np.pi*f) #  spectral density kernel

    #  plot results
    plt.rcParams['figure.figsize'] = (18, 12)
    fig, arx = plt.subplots(2, 1, tight_layout=True, sharex=False, sharey=False)
    arx[0].plot(x, y, lw=2)
    arx[1].plot(F, S)
    arx[0].legend(['data'])
    arx[1].plot(F, Sker)
    arx[1].set_xlim([0, 5000])
    arx[1].set_xlabel('Frequency (Hz)')
    arx[1].legend(['spectral density data', 'spectral density kernel'])
    plt.savefig('../../../../results/figures/isolated_sounds/components/results_' + lfiles[nfile] + '.png')
    plt.close('all')

    #  save results
    np.savez_compressed('../../../../results/files/params_components/params_comp_' + lfiles[nfile],
                        x = x,
                        y = y,
                        fs = fs,
                        F = F,
                        Y = Y,
                        S = S,
                        Nc = Nc,
                        l_param = l,
                        s_param = s,
                        f_param = f)




























#
