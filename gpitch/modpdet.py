import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
import modgp
from scipy.fftpack import fft
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import amtgp


class ModPDet():
    '''Gaussian Process pitch detection using modulated GP'''
    def __init__(self, x, y, fs, ws, jump, Nw=None, bounded=False, whiten=False):
        self.x, self.y = x, y
        self.fs, self.ws = fs, ws # sample rate, and window size (samples)
        self.jump, self.N = jump, x.size
        if Nw == None:
            self.Nw = self.N/self.ws  # number of windows
        else:
            self.Nw = 1
        self.Nh = 10 # number of maximun frequency components in density
        self.bounded = bounded #  bound inducing variables for f_m to be (-1, 1)
        self.whiten = whiten

        # frequency representation data
        self.Y = fft(y.reshape(-1,)) #  FFT data
        self.S =  2./self.N * np.abs(self.Y[0:self.N/2]) #  spectral density data
        self.F = np.linspace(0, fs/2., self.N/2) #  frequency vector

        self.s, self.l, self.f = amtgp.learnparams(X=self.F, S=self.S, Nh=self.Nh) #  Param learning Nh=#harmonics
        sig_scale = 1./ (4.*np.sum(self.s)) #rescale (sigma)
        self.s *= sig_scale

        self.kf = amtgp.Matern12CosineMix(variance=self.s, lengthscale=self.l, period=1./self.f, Nh=self.s.size)
        self.kg = gpflow.kernels.Matern32(input_dim=1, variance=10.*np.random.rand(), lengthscales=10.*np.random.rand())

        self.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)] # split data into windows
        self.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)]
        self.z = self.x_l[0][::jump].copy()

        self.model = modgp.ModGP(self.x_l[0].copy(), self.y_l[0].copy(), self.kf, self.kg, self.z, whiten=self.whiten)
        self.model.likelihood.noise_var = 1e-7
        self.model.likelihood.noise_var.fixed = True
        if self.bounded:
            self.model.q_mu1.transform = gpflow.transforms.Logistic(a=-1.0, b=1.0)
            self.model.q_mu2.transform = gpflow.transforms.Logistic(a=-8.0, b=8.0)
        self.model.kern1.fixed = True # component kernel
        self.model.kern2.fixed = True # activation kernel

        init_list = [None for i in range(0, self.Nw)]  # list to save predictions mean (qm) and variance (qv)
        self.qm1_l = list(init_list)
        self.qm2_l = list(init_list)
        self.qv1_l = list(init_list)
        self.qv2_l = list(init_list)
        self.x_pred_l = list(init_list)
        self.y_pred_l = list(init_list)

        self.qm1 = np.asarray(self.qm1_l).reshape(-1, 1)
        self.qm2 = np.asarray(self.qm2_l).reshape(-1, 1)
        self.qv1 = np.asarray(self.qv1_l).reshape(-1, 1)
        self.qv2 = np.asarray(self.qv2_l).reshape(-1, 1)
        self.x_pred = np.asarray(self.x_pred_l).reshape(-1, 1)
        self.y_pred = np.asarray(self.y_pred_l).reshape(-1, 1)

    def optimize(self, disp, maxiter, reinit_variational_params=True):
        for i in range(self.Nw):
            self.model.X = self.x_l[i].copy()
            self.model.Y = self.y_l[i].copy()
            self.model.Z = self.x_l[i][::self.jump].copy()
            if reinit_variational_params:
                self.model.q_mu1._array = np.zeros(self.model.Z.shape)
                self.model.q_mu2._array = np.zeros(self.model.Z.shape)
                self.model.q_sqrt1._array = np.expand_dims(np.eye(self.model.Z.size), 2)
                self.model.q_sqrt2._array = np.expand_dims(np.eye(self.model.Z.size), 2)
            self.model.optimize(disp=disp, maxiter=maxiter)
            self.qm1_l[i], self.qv1_l[i] = self.model.predict_f(self.x_l[i])
            self.qm2_l[i], self.qv2_l[i] = self.model.predict_g(self.x_l[i])
            self.x_pred_l[i] = self.x_l[i].copy()
            self.y_pred_l[i] = self.y_l[i].copy()

        self.qm1 = np.asarray(self.qm1_l).reshape(-1, 1)
        self.qm2 = np.asarray(self.qm2_l).reshape(-1, 1)
        self.qv1 = np.asarray(self.qv1_l).reshape(-1, 1)
        self.qv2 = np.asarray(self.qv2_l).reshape(-1, 1)
        self.x_pred = np.asarray(self.x_pred_l).reshape(-1, 1)
        self.y_pred = np.asarray(self.y_pred_l).reshape(-1, 1)

    def optimize_restart(self, maxiter, restarts=10):
        init_hyper = [np.zeros((restarts,)) for _ in range(0, 2)] #  save initial hyperparmas values
        learnt_hyper = [np.zeros((restarts,)) for _ in range(0, 2)] #   save learnt hyperparams values
        mse = np.zeros((restarts,)) # save mse error for each restart
        for r in range(0, restarts):
            self.model.kern2.lengthscales = 10.*np.random.rand()
            self.model.kern2.variance = 10.*np.random.rand()
            self.model.whiten = False
            self.model.kern1.fixed = True # component kernel
            self.model.kern2.fixed = False # activation kernel
            init_hyper[0][r] = self.model.kern2.lengthscales.value
            init_hyper[1][r] = self.model.kern2.variance.value
            self.optimize(disp=0, maxiter=maxiter)
            learnt_hyper[0][r] = self.model.kern2.lengthscales.value
            learnt_hyper[1][r] = self.model.kern2.variance.value
            mse[r] = (1./self.N)*np.sum((self.y_pred - amtgp.logistic(self.qm2)*self.qm1)**2)
            print('| len: %8.8f, %8.8f | sig: %8.8f, %8.8f | mse: %8.8f |' % (init_hyper[0][r], learnt_hyper[0][r], init_hyper[1][r], learnt_hyper[1][r], mse[r]) )
        return init_hyper, learnt_hyper, mse


    def plot_results(self, zoom_limits):
        x = self.x_pred
        y = self.y_pred
        fig, fig_array = plt.subplots(3, 2, sharex=False, sharey=False)

        fig_array[0, 0].set_title('Data')
        fig_array[0, 0].plot(x, y, lw=2)
        fig_array[0, 1].set_title('Approximation')
        fig_array[0, 1].plot(x, amtgp.logistic(self.qm2)*self.qm1 , lw=2)

        fig_array[1, 0].set_title('Component')
        fig_array[1, 0].plot(x, self.qm1, color='C0', lw=2)
        fig_array[1, 0].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
                             self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)

        fig_array[1, 1].set_title('Component (zoom in)')
        fig_array[1, 1].plot(x, self.qm1, color='C0', lw=2)
        fig_array[1, 1].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
                             self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)
        fig_array[1, 1].set_xlim(zoom_limits)

        fig_array[2, 0].set_title('Activation')
        fig_array[2, 0].plot(x, amtgp.logistic(self.qm2), color='g', lw=2)
        fig_array[2, 0].fill_between(x[:, 0], amtgp.logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
                             amtgp.logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)

        fig_array[2, 1].set_title('Activation (zoom in)')
        fig_array[2, 1].plot(x, amtgp.logistic(self.qm2), 'g', lw=2)
        fig_array[2, 1].fill_between(x[:, 0], amtgp.logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
                             amtgp.logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)
        fig_array[2, 1].set_xlim(zoom_limits)

        return fig, fig_array

    def plot_learntprior(self, arg):
        pass
        # xsample = np.linspace(0, 0.05, 800).reshape(-1,1)
        # Kf = m.model.kern1.compute_K_symm(xsample)
        # Kg = m.model.kern2.compute_K_symm(xsample)
        #
        # print m.model.kern2.lengthscales.value
        # print m.model.kern2.variance.value
        #
        # # sample prom learnt priors
        # sample_f = np.random.multivariate_normal(np.zeros(xsample.size), Kf, 2).T
        # sample_g = np.random.multivariate_normal(np.zeros(xsample.size), Kg, 5).T
        #
        # plt.figure(), plt.plot(m.x, m.y)
        # plt.figure(), plt.plot(m.F, m.S)














#
