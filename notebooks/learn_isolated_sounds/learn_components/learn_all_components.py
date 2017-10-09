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


gpitch.amtgp.init_settings(visible_device = '0', interactive=True) #  configure gpu usage and plotting
location = "../../../../../../datasets/MAPS/AkPnBcht/ISOL/NO/sample_rate_16khz/" # load list of files no analyse
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
    np.savez_compressed('../../../results/isolated_sounds/params_comp_' + lfiles[nfile],
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
