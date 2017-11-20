import sys
import os
import pickle
import numpy as np
import tensorflow as tf
import gpflow
sys.path.append('../../../../../')
import gpitch
from gpitch.amtgp import logistic
from gpitch import myplots


visible_device = sys.argv[1]  # configure gpu usage
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True)

saveloc = '../../../../../../results/files/svi/script/'
dirloc = '../../../../../../datasets/maps/sample_rate_16khz/'
pattern = '*F*.wav'
bounds = [21, 109]
midi = np.asarray([str(i) for i in range(bounds[0], bounds[1])]).reshape(-1,)
filel = gpitch.amtgp.load_filenames(directory=dirloc, pattern=pattern, bounds=bounds)
Np = filel.size  # number of pitches to analyze

fs = 16e3  # sample frequency
N = 16000  # number of data points to load
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time vector
Nc = 10  #  number of harmonics for component kernels
dec = 160  # decimation factor
minibatch_size = 200  # batch size svi
maxiter = 2000  #  maximun number of iterations in optimization
learning_rate = 0.01  # learning rate svi optimization
maxh = 20  # max number of harmoncis in component kernel

Y = np.zeros((N, Np))  # matrix with all training data
ideal_f0 = np.zeros((Np,1))  # vector ideal natural frequency each pitch
Fs = [None]*Np  # list of harmonic array for each pitch comp kernel
Ss = [None]*Np  # list of corresponding variances array for each pitch comp kernel
F  = np.zeros((N//2, 1))  # frequency vector
S  = np.zeros((N//2, Np)) # matrix spectral density each trainig data signal
thres = np.zeros((Np, 1))  # threshold used for peak selection for each pitch

for i in range(Np):  # load data
    y, fs = gpitch.amtgp.wavread(dirloc + filel[i], start=5000, N=N)  # load data
    Y[:, i] = y.reshape(-1,).copy()
    ideal_f0[i] = gpitch.amtgp.midi2frec(int(midi[i]))

    Fs[i], Ss[i], F[:, 0], S[:, i], thres[i] = gpitch.amtgp.init_com_params(y=y, fs=fs,
                                                                            maxh=maxh,
                                                                            ideal_f0=
                                                                            ideal_f0[i],
                                                                            scaled=True,
                                                                            win_size=10)
for i in range(Np):
    y = Y[:, i].reshape(-1,1).copy()
    z = np.vstack((x[::dec].copy(), x[-1].copy()))

    kern_com = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=0.1, variances=Ss[i],
                                            frequencies=Fs[i], Nc=Fs[i].size)
    kern_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.25, variance=10.)

    m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=kern_com, kern_act=kern_act, whiten=True,
                           minibatch_size=minibatch_size)

    m.fixed_msmkern_params(freq=False, var=True)
    m.kern_com.lengthscales.fixed = False
    m.kern_com.lengthscales.transform = gpflow.transforms.Logistic(0., 10.0)
    m.kern_act.fixed = False
    m.likelihood.variance.fixed = False
    m.z.fixed = True
    print('learning from ' + filel[i])
    m.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)
    pickle.dump(m, open(saveloc + "maps_pitch_" + midi[i] + ".p", "wb"))
    tf.reset_default_graph()
