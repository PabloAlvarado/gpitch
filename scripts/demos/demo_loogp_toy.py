import time, sys, os
sys.path.append('../../')
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


visible_device = sys.argv[1]  # which gpu to use
init_model = sys.argv[2].lower() == '1'  # if true (1) initialize the gpflow model, otherwise reuse existing model
loc = '../../../results/files/demos/loogp/'  # location save results
if init_model:
    #np.random.seed(29)
    gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True) #  confi gpu usage, plot
    fs, N = 16e3, 1600  # generate synthetic data, number of samples
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time
    noise_var = 1.e-3  # noise variance
    pitch1, pitch2 = 440.00, 659.25  # A4 (Hz), E5 (Hz)
    leng = np.asarray([0.25, 0.25, 0.25]).reshape(-1,)
    var = np.asarray([0.25, 0.2, 0.15]).reshape(-1,)
    var_rescale = 1./ (4.*np.sum(var)) #rescale (sigma)
    var *= var_rescale
    freq1 = np.asarray([pitch1, 2*pitch1, 3*pitch1]).reshape(-1,)
    freq2 = np.asarray([pitch2, 2*pitch2, 3*pitch2]).reshape(-1,)
    kenv1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
    kenv2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.005, variance=10.)
    kper1 = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=leng, variances=var, frequencies=freq1)
    kper2 = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=leng, variances=var, frequencies=freq2)
    Kenv1 = kenv1.compute_K_symm(x)
    Kenv2 = kenv2.compute_K_symm(x)
    Kper1 = kper1.compute_K_symm(x)
    Kper2 = kper2.compute_K_symm(x)
    f1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper1).reshape(-1, 1)
    f2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper2).reshape(-1, 1)
    f1 /= np.max(np.abs(f1))
    f2 /= np.max(np.abs(f2))
    g1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv1).reshape(-1, 1)
    g2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv2).reshape(-1, 1)
    source1 = gpitch.amtgp.logistic(g1)*f1
    source2 = gpitch.amtgp.logistic(g2)*f2
    mean = source1 + source2
    y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)
    maxiter, dec, ws = 250, 10, N//2  # maxiter, decimation factor, window size in samples
    kc, ka = [kper1, kper2], [kenv1, kenv2]
    model = gpitch.loopdet.LooPDet(x=x, y=y, kern_comps=kc, kern_acts=ka, ws=ws, dec=dec, whiten=True)
    model.m.likelihood.noise_var.fixed = True
    model.m.kern_f1.fixed = True
    model.m.kern_f2.fixed = True
    model.m.kern_g1.fixed = True
    model.m.kern_g2.fixed = True
else:
    np.random.seed()  # sample other toy examples without recompile the graph
    f1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper1).reshape(-1, 1)
    f2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper2).reshape(-1, 1)
    f1 /= np.max(np.abs(f1))
    f2 /= np.max(np.abs(f2))
    g1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv1).reshape(-1, 1)
    g2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv2).reshape(-1, 1)
    source1 = gpitch.amtgp.logistic(g1)*f1
    source2 = gpitch.amtgp.logistic(g2)*f2
    mean = source1 + source2
    y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)
    model.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, model.Nw)] # split data into windows
    model.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, model.Nw)]

model.m.likelihood.noise_var = noise_var
model.optimize_windowed(disp=1, maxiter=maxiter)
model.save_results(loc + 'results_toy')
np.savez_compressed(loc + 'data_toy', f1=f1, f2=f2, g1=g1, g2=g2)


































#
