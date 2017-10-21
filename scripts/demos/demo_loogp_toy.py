import time, sys, os
sys.path.append('../../')
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


visible_device = sys.argv[1]  # which gpu to use
init_model = sys.argv[2].lower() == 'true'  # if true initialize the gpflow model, otherwise reuse existent model
if init_model:
    np.random.seed(29)
    gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True) #  confi gpu usage, plot
    fs = 16e3  # generate synthetic data
    N = 100  # number of samples
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time
    noise_var = 1.e-3  # noise variance
    pitch1 = 440.00  # Hertz, A4 (La)
    pitch2 = 659.25  # Hertz, E5 (Mi)
    kenv1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
    kenv2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.005, variance=10.)
    kper1 = gpflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                          variance=np.sqrt(0.5), period=1./pitch1)
    kper2 = gpflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                          variance=np.sqrt(0.5), period=1./pitch2)
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

    maxiter, dec, ws = 100, 10, N  # maxiter, decimation factor, window size in samples
    kc, ka = [kper1, kper2], [kenv1, kenv2]

    model = gpitch.loopdet.LooPDet(x=x, y=y, kern_comps=kc, kern_acts=ka, ws=ws, dec=dec, whiten=True)
    model.m.likelihood.noise_var.fixed = True
    model.m.kern_f1.fixed = True
    model.m.kern_f2.fixed = True
    model.m.kern_g1.fixed = True
    model.m.kern_g2.fixed = True
model.m.likelihood.noise_var = noise_var
model.optimize_windowed(disp=1, maxiter=maxiter)
model.save_results('../../../results/files/demos/loogp/results_toy')


































#
