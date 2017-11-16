import sys
sys.path.append('../../../../')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow, gpitch
reload(gpitch)
from gpitch.amtgp import logistic
from scipy.fftpack import fft, ifft, ifftshift
from scipy import signal
import time
from gpitch import myplots
import soundfile
import pickle

plt.close('all')
plt.interactive(True)
plt.rcParams['figure.figsize'] = (16, 5)  # set plot size
pitchlist = np.asarray(['60', '64', '67', '72', '76'])
Np = pitchlist.size  # number of pitches to analyze

for i in range(Np):
    midi = pitchlist[i]
    filename = '../../../../../datasets/maps/' + str(midi) + '_1-down.wav'
    N = 16000 # number of data points to load
    #y, fs = soundfile.read(filename, frames=N)  # Load data
    fs, y = gpitch.amtgp.wavread(filename, start=5000, N=N) # load 2 secs of data
    y = y.reshape(-1,1)
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)

    # Define model
    Nc = 10
    ideal_f0 = gpitch.amtgp.midi2frec(int(midi))
    F_star, S_star, F, Y, S = gpitch.amtgp.init_com_params(y=y, fs=fs, Nh=Nc, ideal_f0=ideal_f0, scaled=True)
    # Define kernels for component and activation, and generate model object ("sigmoid model")
    kern_com = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=0.1, variances=S_star,
                                            frequencies=F_star, Nc=Nc)
    kern_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=1., variance=10.)
    dec = 160  # decimation factor
    minibatch_size = 200
    z = np.vstack((x[::dec].copy(), x[-1].copy()))
    m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=kern_com, kern_act=kern_act,
                           whiten=True,  minibatch_size=minibatch_size)
    # Set all parameters free to optimize, but variances of component
    m.kern_com.fixed = True
    m.kern_com.lengthscales.fixed = False
    m.kern_com.lengthscales.transform = gpflow.transforms.Logistic(0., 10.0)
    m.fixed_msmkern_params(freq=False, var=True)
    m.kern_act.fixed = False
    m.likelihood.variance.fixed = False
    m.z.fixed = True

    st = time.time()
    logt = []
    logx = []
    logf = []
    def logger(x):
        if (logger.i % 10) == 0:
            logx.append(x)
            logf.append(m._objective(x)[0])
            logt.append(time.time() - st)
        logger.i+=1
    logger.i = 1
    m.x.minibatch_size = minibatch_size
    m.y.minibatch_size = minibatch_size

    maxiter = 2000
    m.optimize(method=tf.train.AdamOptimizer(learning_rate=0.01), maxiter=maxiter,
               callback=logger)


    m.logf = logf
    pickle.dump(m, open("save_model_maps5_pitch_" + midi + ".p", "wb"))

    tf.reset_default_graph()
