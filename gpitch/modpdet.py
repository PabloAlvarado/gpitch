import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
import modgp
from scipy.fftpack import fft
import os
import tensorflow as tf
import amtgp
import kernels

class ModPDet():
    '''Gaussian Process pitch detection using modulated GP'''

    def __init__(self, x, y, kern_com, kern_act, ws, dec, Nw=None, bounded=False, whiten=False):
        self.x, self.y = x, y
        self.ws = ws # sample rate, and window size (samples)
        self.dec, self.N = dec, x.size
        if Nw == None: self.Nw = self.N/self.ws  # number of windows
        else: self.Nw = 1
        self.bounded = bounded #  bound inducing variables for f_m to be (-1, 1)
        self.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)] # split data into windows
        self.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)]
        #self.z = self.x_l[0][::dec].copy()
        self.z = np.vstack(( self.x_l[0][::dec].copy(), self.x_l[0][-1].copy() ))
        self.model = modgp.ModGP(self.x_l[0].copy(), self.y_l[0].copy(), kern_com, kern_act, self.z, whiten=whiten)
        self.model.likelihood.noise_var.transform = gpflow.transforms.Logistic(a=0., b=1e-1)

        if self.bounded:
            self.model.q_mu1.transform = gpflow.transforms.Logistic(a=-1.0, b=1.0)
            self.model.q_mu2.transform = gpflow.transforms.Logistic(a=-8.0, b=8.0)

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

    def optimize_windowed(self, disp, maxiter, init_zero=True):
        for i in range(self.Nw):
            self.model.X = self.x_l[i].copy()
            self.model.Y = self.y_l[i].copy()
            #self.model.Z = self.x_l[i][::self.dec].copy()
            self.model.Z = np.vstack(( self.x_l[i][::self.dec].copy(), self.x_l[i][-1].copy() ))
            if init_zero:
                self.model.q_mu1 = np.zeros(self.model.Z.shape)
                self.model.q_mu2 = np.zeros(self.model.Z.shape)
                self.model.q_sqrt1 = np.expand_dims(np.eye(self.model.Z.size), 2)
                self.model.q_sqrt2 = np.expand_dims(np.eye(self.model.Z.size), 2)

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
        init_hyper = [np.zeros((restarts,)) for _ in range(0, 5)] #  save initial hyperparmas values
        learnt_hyper = [np.zeros((restarts,)) for _ in range(0, 5)] #   save learnt hyperparams values
        mse = np.zeros((restarts,)) # save mse error for each restart
        for r in range(0, restarts):

            self.model.kern1.lengthscales = 0.1*np.random.rand()
            #for i in range(self.model.kern1.Nc): # generate a param object for each  var, and freq, they must be (Nc,) arrays.
            #    setattr(self.model.kern1, 'variance_' + str(i+1), 0.25*np.random.rand() )
            #    setattr(self.model.kern1, 'frequency_' + str(i+1), (i+1)*self.model.kern1.ideal_f0 + np.sqrt(5)*np.random.randn()  )

            self.model.kern2.lengthscales = 1.*np.random.rand()
            self.model.kern2.variance = 15.*np.random.rand()

            self.model.likelihood.noise_var = 0.1*np.random.rand()

            init_hyper[0][r] = self.model.kern2.lengthscales.value
            init_hyper[1][r] = self.model.kern2.variance.value
            init_hyper[2][r] = self.model.likelihood.noise_var.value
            init_hyper[3][r] = self.model.kern1.lengthscales.value
            # init_hyper[0][r] = self.model.kern1.variance_1.value
            # init_hyper[1][r] = self.model.kern1.variance_2.value
            # init_hyper[2][r] = self.model.kern1.variance_3.value
            # init_hyper[3][r] = self.model.kern1.variance_4.value
            # init_hyper[4][r] = self.model.kern1.variance_5.value
            self.optimize_windowed(disp=1, maxiter=maxiter)
            # learnt_hyper[0][r] = self.model.kern1.variance_1.value
            # learnt_hyper[1][r] = self.model.kern1.variance_2.value
            # learnt_hyper[2][r] = self.model.kern1.variance_3.value
            # learnt_hyper[3][r] = self.model.kern1.variance_4.value
            # learnt_hyper[4][r] = self.model.kern1.variance_5.value
            learnt_hyper[0][r] = self.model.kern2.lengthscales.value
            learnt_hyper[1][r] = self.model.kern2.variance.value
            learnt_hyper[2][r] = self.model.likelihood.noise_var.value
            learnt_hyper[3][r] = self.model.kern1.lengthscales.value
            mse[r] = (1./self.N)*np.sum((self.y_pred - amtgp.logistic(self.qm2)*self.qm1)**2)
            print('| len: %4.4f, %4.4f | sig: %4.4f, %4.4f | noise_var: %4.4f, %4.4f | l_com: %4.4f, %4.4f |' % (init_hyper[0][r], learnt_hyper[0][r],
                                                                                                                 init_hyper[1][r], learnt_hyper[1][r],
                                                                                                                 init_hyper[2][r], learnt_hyper[2][r],
                                                                                                                 init_hyper[3][r], learnt_hyper[3][r]) )
            #print('| v1: %4.4f, %4.4f | v2: %4.4f, %4.4f | v3: %4.4f, %4.4f | v4: %4.4f, %4.4f | v5: %4.4f, %4.4f | ' % (init_hyper[0][r], learnt_hyper[0][r],
            #                                                                                                             init_hyper[1][r], learnt_hyper[1][r],
        #                                                                                                                 init_hyper[2][r], learnt_hyper[2][r],
        #                                                                                                                 init_hyper[3][r], learnt_hyper[3][r],
        #                                                                                                                 init_hyper[4][r], learnt_hyper[4][r]))
        return init_hyper, learnt_hyper, mse


    def plot_results(self, zoom_limits):
        pass
        # x = self.x_pred
        # y = self.y_pred
        # fig, fig_array = plt.subplots(3, 2, sharex=False, sharey=False)
        #
        # fig_array[0, 0].set_title('Data')
        # fig_array[0, 0].plot(x, y, lw=2)
        # fig_array[0, 1].set_title('Approximation')
        # fig_array[0, 1].plot(x, amtgp.logistic(self.qm2)*self.qm1 , lw=2)
        #
        # fig_array[1, 0].set_title('Component')
        # fig_array[1, 0].plot(x, self.qm1, color='C0', lw=2)
        # fig_array[1, 0].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
        #                      self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)
        #
        # fig_array[1, 1].set_title('Component (zoom in)')
        # fig_array[1, 1].plot(x, self.qm1, color='C0', lw=2)
        # fig_array[1, 1].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
        #                      self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)
        # fig_array[1, 1].set_xlim(zoom_limits)
        #
        # fig_array[2, 0].set_title('Activation')
        # fig_array[2, 0].plot(x, amtgp.logistic(self.qm2), color='g', lw=2)
        # fig_array[2, 0].fill_between(x[:, 0], amtgp.logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
        #                      amtgp.logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)
        #
        # fig_array[2, 1].set_title('Activation (zoom in)')
        # fig_array[2, 1].plot(x, amtgp.logistic(self.qm2), 'g', lw=2)
        # fig_array[2, 1].fill_between(x[:, 0], amtgp.logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
        #                      amtgp.logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)
        # fig_array[2, 1].set_xlim(zoom_limits)
        #
        # return fig, fig_array

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
