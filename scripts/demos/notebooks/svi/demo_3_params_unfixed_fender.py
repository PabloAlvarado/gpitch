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

plt.close('all')
plt.interactive(True)
plt.rcParams['figure.figsize'] = (16, 5)  # set plot size
pitchlist = np.asarray(['60', '64', '67'])
Np = pitchlist.size  # number of pitches to analyze

for i in range(Np):
    midi = pitchlist[i]
    filename = '../../../../../datasets/fender/train/m_' + str(midi) + '.wav'
    N = 3200 # number of data points to load
    y, fs = soundfile.read(filename, frames=N)  # Load data
    y = y.reshape(-1,1)
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
    plt.figure()
    plt.title('data with pitch ' + midi)
    plt.plot(x, y, lw=2)

    # Define model
    Nc = 10
    ideal_f0 = gpitch.amtgp.midi2frec(int(midi))
    F_star, S_star, F, Y, S = gpitch.amtgp.init_com_params(y=y, fs=fs, Nh=Nc, ideal_f0=ideal_f0, scaled=True)
    plt.figure()
    plt.plot(F, S/np.max(S))
    plt.plot(F_star, S_star/np.max(S_star), 'xk', mew=2)
    plt.legend(['Smoothed spectral density of pitch ' + midi, 'location of harmonics found for initialization'])
    plt.xlim([0, 8000])
    # Define kernels for component and activation, and generate model object ("sigmoid model")
    kern_com = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=0.1, variances=S_star,
                                            frequencies=F_star, Nc=Nc)
    kern_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=1., variance=10.)
    dec = 120  # decimation factor
    z = np.vstack((x[::dec].copy(), x[-1].copy()))
    m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=kern_com, kern_act=kern_act, whiten=True)
    # Set all parameters free to optimize, but variances of component
    m.kern_com.fixed = True
    m.kern_com.lengthscales.fixed = False
    m.kern_com.lengthscales.transform = gpflow.transforms.Logistic(0., 1.0)
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
    m.x.minibatch_size = 100
    m.y.minibatch_size = 100

    maxiter = 1000
    m.optimize(method=tf.train.AdamOptimizer(learning_rate=0.01), maxiter=maxiter,
               callback=logger)

    plt.figure()
    plt.plot(-np.array(logf))
    plt.xlabel('iteration')
    plt.ylabel('ELBO')

    mean_f, var_f, mean_g, var_g, x_plot  = m.predict_all(x)

    myplots.plot_results(mean_f, var_f, mean_g, var_g, x_plot, y, z, xlim=[0.0, 0.2])

    k_plot_model = m.kern_com.compute_K(x, np.asarray(0.).reshape(-1,1))


    Yk1 = fft(k_plot_model.reshape(-1,)) #  FFT data
    Sk1 =  2./N * np.abs(Yk1[0:N//2]) #  spectral density data

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(F, S / np.max(np.abs(S)), lw=2)
    plt.legend([' Spectral density data'])
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 8000])

    plt.subplot(2, 1, 2)
    plt.plot(F, S / np.max(np.abs(S)), 'r', lw=2)
    plt.plot(F, Sk1 / np.max(np.abs(Sk1)), lw=2)
    plt.legend([' Spectral density learned component kernel'])
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 8000])


    xkernel = np.linspace(0,3, 48000).reshape(-1, 1)
    k_plot_model = m.kern_com.compute_K(xkernel, np.asarray(0.).reshape(-1,1))
    plt.figure(figsize=(16, 4))
    plt.plot(xkernel, k_plot_model, lw=2)

    tf.reset_default_graph()
