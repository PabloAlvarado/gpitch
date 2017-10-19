import time, sys, os
sys.path.append('../../')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


class LooPDet():
    '''Leave one out pitch detector'''
    def __init__(self, x, y, kern_comps, kern_acts, ws, dec, Nw=None, bounded=False, whiten=False):
        '''
        x : time vector
        y : audio mixture
        ws : number of samples per windows (window width)
        dec : decimation factor for inducing variables
        Nw : number of windows to analyze
        bounded : option to set limits to variational parameters
        whiten : gpflow option
        '''
        self.x, self.y = x, y  # data
        self.ws = ws  # analysis window size (samples)
        self.dec = dec
        self.N = x.size
        if Nw == None: self.Nw = int(self.N//self.ws)  # number of windows
        else: self.Nw = Nw
        self.Nh = 10 # number of maximun frequency components in density
        self.bounded = bounded #  bound inducing variables for f_m to be (-1, 1)
        self.whiten = whiten

        self.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)] # split data into windows
        self.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)]

        self.qm1_l, self.qm2_l, self.qm3_l, self.qm4_l = [[None]*self.Nw for _ in range(4)]
        self.qv1_l, self.qv2_l, self.qv3_l, self.qv4_l = [[None]*self.Nw for _ in range(4)]
        self.x_pred_l, self.y_pred_l = [None]*self.Nw, [None]*self.Nw
        self.qm1 = np.zeros(self.Nw * self.ws)
        self.qm2 = self.qm1.copy()
        self.qm3 = self.qm1.copy()
        self.qm4 = self.qm1.copy()
        self.qv1 = np.ones(self.Nw * self.ws)
        self.qv2 = self.qv1.copy()
        self.qv3 = self.qv1.copy()
        self.qv4 = self.qv1.copy()
        self.x_pred = self.qm1.copy()
        self.y_pred = self.qm1.copy()
        self.yhat = self.qm1.copy()

        self.z = self.x_l[0][::dec].copy()  # inducing variables
        self.m = gpitch.loogp.LooGP(self.x_l[0].copy(), self.y_l[0].copy(), [kern_comps[0], kern_comps[1]],
                                    [kern_acts[0], kern_acts[1]], self.z, whiten=self.whiten)  # init model
        self.m.likelihood.noise_var = 1e-4
        self.m.likelihood.noise_var.fixed = True
        self.m.kern_f1.fixed = True
        self.m.kern_f2.fixed = True
        self.m.kern_g1.fixed = True
        self.m.kern_g2.fixed = True

    def optimize_windowed(self, disp, maxiter):
        # pass
        for i in range(self.Nw):
            self.m.X = self.x_l[i].copy()
            self.m.Y = self.y_l[i].copy()
            self.m.Z = self.x_l[i][::self.dec].copy()

            self.m.q_mu1._array = np.zeros(self.m.Z.shape)
            self.m.q_mu2._array = np.zeros(self.m.Z.shape)
            self.m.q_mu3._array = np.zeros(self.m.Z.shape)
            self.m.q_mu4._array = np.zeros(self.m.Z.shape)

            self.m.q_sqrt1._array = np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt2._array = np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt3._array = np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt4._array = np.expand_dims(np.eye(self.m.Z.size), 2)

            self.m.optimize(disp=disp, maxiter=maxiter)

            self.qm1_l[i], self.qv1_l[i] = self.m.predict_f1(self.x_l[i])
            self.qm2_l[i], self.qv2_l[i] = self.m.predict_g1(self.x_l[i])
            self.qm3_l[i], self.qv3_l[i] = self.m.predict_f2(self.x_l[i])
            self.qm4_l[i], self.qv4_l[i] = self.m.predict_g2(self.x_l[i])
            self.x_pred_l[i] = self.x_l[i].copy()
            self.y_pred_l[i] = self.y_l[i].copy()

        self.qm1 = np.asarray(self.qm1_l).reshape(-1, )
        self.qm2 = np.asarray(self.qm2_l).reshape(-1, )
        self.qv1 = np.asarray(self.qv1_l).reshape(-1, )
        self.qv2 = np.asarray(self.qv2_l).reshape(-1, )
        self.x_pred = np.asarray(self.x_pred_l).reshape(-1, )
        self.y_pred = np.asarray(self.y_pred_l).reshape(-1, )
        self.yhat = logistic(self.qm2)*self.qm1 + logistic(self.qm4)*self.qm3



    def plot_results(self):
        nrows, ncols = 4, 2
        plt.figure(figsize=(ncols*18, nrows*6))
        plt.subplot(nrows, ncols, (1, 2))
        plt.title('data and prediction')
        plt.plot(self.x_pred, self.y_pred, '.k', mew=1)
        plt.plot(self.x_pred, self.yhat , lw=2)

        plt.subplot(nrows, ncols, 3)
        plt.title('component 1')
        plt.plot(self.x_pred, self.qm1, color='C0', lw=2)
        plt.fill_between(self.x_pred, self.qm1-2*np.sqrt(self.qv1), self.qm1+2*np.sqrt(self.qv1),
                         color='C0', alpha=0.2)

        plt.subplot(nrows, ncols, 4)
        plt.title('component 2')
        plt.plot(self.x_pred, self.qm3, color='C0', lw=2)
        plt.fill_between(self.x_pred, self.qm3-2*np.sqrt(self.qv3), self.qm3+2*np.sqrt(self.qv3),
                         color='C0', alpha=0.2)

        plt.subplot(nrows, ncols, 5)
        plt.title('activation 1')
        plt.plot(self.x_pred, logistic(self.qm2), 'g', lw=2)
        plt.fill_between(self.x_pred, logistic(self.qm2-2*np.sqrt(self.qv2)),
                         logistic(self.qm2+2*np.sqrt(self.qv2)), color='g', alpha=0.2)

        plt.subplot(nrows, ncols, 6)
        plt.title('activation 2')
        plt.plot(self.x_pred, logistic(self.qm4), 'g', lw=2)
        plt.fill_between(self.x_pred, logistic(self.qm4 - 2*np.sqrt(self.qv4)),
                         logistic(self.qm4+2*np.sqrt(self.qv4)), color='g', alpha=0.2)

        plt.subplot(nrows, ncols, 7)
        plt.title('source 1')
        plt.plot(self.x_pred, logistic(self.qm2)*self.qm1, lw=2)

        plt.subplot(nrows, ncols, 8)
        plt.title('source 2')
        plt.plot(self.x_pred, logistic(self.qm4)*self.qm3, lw=2)
