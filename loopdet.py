import time, sys, os
sys.path.append('../../')
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic, init_com_params


class LooPDet():
    '''Leave one out pitch detector'''
    def __init__(self, x, y, ker_com, ker_act, plist, ws, dec, Nw=None, bounded=False, whiten=True):
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
        self.plist = plist  # list of pitches to detect

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
        self.m = gpitch.loogp.LooGP(self.x_l[0].copy(), self.y_l[0].copy(), [ker_com[0], ker_com[1]],
                                    [ker_act[0], ker_act[1]], self.z, whiten=self.whiten)  # init model
        self.m.likelihood.noise_var = 1e-3
        self.nrows, self.ncols = 4, 2  # number of rows and columns when plotting results

    def optimize_windowed(self, disp, maxiter):
        '''
        This function call the gpflow optimizer for every window of analysis
        '''
        for i in range(self.Nw):
            self.m.X = self.x_l[i].copy()
            self.m.Y = self.y_l[i].copy()
            self.m.Z = self.x_l[i][::self.dec].copy()
            self.m.q_mu1 = 2.*self.y_l[i][::self.dec].copy()
            self.m.q_mu2 = np.zeros(self.m.Z.shape)
            self.m.q_mu3 = 2.*self.y_l[i][::self.dec].copy()
            self.m.q_mu4 = np.zeros(self.m.Z.shape)
            self.m.q_sqrt1 = 1e-3*np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt2 = 1e-3*np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt3 = 1e-3*np.expand_dims(np.eye(self.m.Z.size), 2)
            self.m.q_sqrt4 = 1e-3*np.expand_dims(np.eye(self.m.Z.size), 2)

            self.m.optimize(disp=disp, maxiter=maxiter)
            self.qm1_l[i], self.qv1_l[i] = self.m.predict_f1(self.x_l[i])
            self.qm2_l[i], self.qv2_l[i] = self.m.predict_g1(self.x_l[i])
            self.qm3_l[i], self.qv3_l[i] = self.m.predict_f2(self.x_l[i])
            self.qm4_l[i], self.qv4_l[i] = self.m.predict_g2(self.x_l[i])
            self.x_pred_l[i] = self.x_l[i].copy()
            self.y_pred_l[i] = self.y_l[i].copy()

        self.qm1 = np.asarray(self.qm1_l).reshape(-1, )
        self.qm2 = np.asarray(self.qm2_l).reshape(-1, )
        self.qm3 = np.asarray(self.qm3_l).reshape(-1, )
        self.qm4 = np.asarray(self.qm4_l).reshape(-1, )

        self.qv1 = np.asarray(self.qv1_l).reshape(-1, )
        self.qv2 = np.asarray(self.qv2_l).reshape(-1, )
        self.qv3 = np.asarray(self.qv3_l).reshape(-1, )
        self.qv4 = np.asarray(self.qv4_l).reshape(-1, )

        self.x_pred = np.asarray(self.x_pred_l).reshape(-1, )
        self.y_pred = np.asarray(self.y_pred_l).reshape(-1, )
        self.yhat = logistic(self.qm2)*self.qm1 + logistic(self.qm4)*self.qm3


    def save_results(self, filename):
        np.savez_compressed(filename,
                            x_pred = self.x_pred,
                            y_pred = self.y_pred,
                            yhat = self.yhat,
                            qm1 = self.qm1,
                            qm2 = self.qm2,
                            qm3 = self.qm3,
                            qm4 = self.qm4,
                            qv1 = self.qv1,
                            qv2 = self.qv2,
                            qv3 = self.qv3,
                            qv4 = self.qv4)


    def update_params(self, params):
        ''''
        update parameters of graph
        '''
        # init all param values to zero or one
        for i in range(self.m.kern_f1.Nc):
            setattr(self.m.kern_f1, 'variance_' + str(i+1), 0.)
            setattr(self.m.kern_f1, 'lengthscale_' + str(i+1), 1.)
            setattr(self.m.kern_f1, 'frequency_' + str(i+1), 0.)

        for i in range(self.m.kern_f2.Nc):
            setattr(self.m.kern_f2, 'variance_' + str(i+1), 0.)
            setattr(self.m.kern_f2, 'lengthscale_' + str(i+1), 1.)
            setattr(self.m.kern_f2, 'frequency_' + str(i+1), 0.)

        # upload new params values
        for i in range(params['s_com1'].size):
            setattr(self.m.kern_f1, 'variance_' + str(i+1), params['s_com1'][i])
            setattr(self.m.kern_f1, 'lengthscale_' + str(i+1), params['l_com1'][i])
            setattr(self.m.kern_f1, 'frequency_' + str(i+1), params['f_com1'][i])

        for i in range(params['s_com2'].size):
            setattr(self.m.kern_f2, 'variance_' + str(i+1), params['s_com2'][i])
            setattr(self.m.kern_f2, 'lengthscale_' + str(i+1), params['l_com2'][i])
            setattr(self.m.kern_f2, 'frequency_' + str(i+1), params['f_com2'][i])

        setattr(self.m.kern_g1, 'variance', params['s_act1'])
        setattr(self.m.kern_g1, 'lengthscales', params['l_act1'])

        setattr(self.m.kern_g2, 'variance', params['s_act2'])
        setattr(self.m.kern_g2, 'lengthscales', params['l_act2'])


    def learnparams(self, xtrain, ytrain, fs, Nh, run_training=True):
        Np = ytrain.shape[1]  # number of pitches to learn
        params = [None]*Np
        if run_training:
            for i in range(Np):
                ideal_f0 = gpitch.amtgp.midi2frec(int(self.plist[i]))  # Init comp params 4 each pitch
                params[i] = init_com_params(y=ytrain[:,i], fs=fs, Nh=Nh, ideal_f0=ideal_f0, win_size=6)


                # m.model.kern1.fixed = True
                # m.model.kern1.lengthscales.fixed = False
                # m.model.kern1.lengthscales.transform = gpflow.transforms.Logistic(0., 0.1)
                # m.model.kern1.frequency_1.fixed = False
                # m.model.kern1.frequency_2.fixed = False
                # m.model.kern1.frequency_3.fixed = False
                # m.model.kern1.frequency_4.fixed = False
                # m.model.kern1.frequency_5.fixed = False
                # m.model.kern1.frequency_6.fixed = False
                # m.model.kern1.frequency_7.fixed = False
                # m.model.kern1.frequency_8.fixed = False
                # m.model.kern1.frequency_9.fixed = False
                # m.model.kern1.frequency_10.fixed = False
                # m.model.kern1.frequency_11.fixed = False
                # m.model.kern1.frequency_12.fixed = False
                # m.model.kern1.frequency_13.fixed = False
                # m.model.kern1.frequency_14.fixed = False
                # m.model.kern1.frequency_15.fixed = False
                # m.model.kern2.fixed = False
                # m.model.likelihood.noise_var.fixed = False
                #
                # maxiter, restarts = 500, 3
                # init_hyper, learnt_hyper, mse = m.optimize_restart(maxiter=maxiter, restarts=restarts)
                #
                # m.model.kern2.lengthscales = learnt_hyper[0].mean().copy()
                # m.model.kern2.variance = learnt_hyper[1].mean().copy()
                # m.model.likelihood.noise_var = learnt_hyper[2].mean().copy()
                # m.model.kern1.lengthscales = learnt_hyper[3].mean().copy()
                # m.model.optimize(disp=1, maxiter=250)
                #
                # m.model.kern1.fixed = True
                # m.model.kern2.fixed = True
                # m.model.likelihood.noise_var.fixed = True
                #
                # m.model.kern1.variance_1.fixed = False
                # m.model.kern1.variance_2.fixed = False
                # m.model.kern1.variance_3.fixed = False
                # m.model.kern1.variance_4.fixed = False
                # m.model.kern1.variance_5.fixed = False
                # m.model.kern1.variance_6.fixed = False
                # m.model.kern1.variance_7.fixed = False
                # m.model.kern1.variance_8.fixed = False
                # m.model.kern1.variance_9.fixed = False
                # m.model.kern1.variance_10.fixed = False
                # m.model.kern1.variance_11.fixed = False
                # m.model.kern1.variance_12.fixed = False
                # m.model.kern1.variance_13.fixed = False
                # m.model.kern1.variance_14.fixed = False
                # m.model.kern1.variance_15.fixed = False
                #
                # m.model.optimize(disp=1, maxiter=10)
        else:
            pass  # load learned parameters
        return params



    def plot_results(self):
        '''
        Plot infered components and activations
        '''
        pass
        # plt.figure(figsize=(self.ncols*18, self.nrows*6))
        # plt.subplot(self.nrows, self.ncols, (1, 2))
        # plt.title('data and prediction')
        # plt.plot(self.x_pred, self.y_pred, '.k', mew=1)
        # plt.plot(self.x_pred, self.yhat , lw=2)
        #
        # plt.subplot(self.nrows, self.ncols, 3)
        # plt.title('source 1')
        # plt.plot(self.x_pred, logistic(self.qm2)*self.qm1, lw=2)
        #
        # plt.subplot(self.nrows, self.ncols, 4)
        # plt.title('source 2')
        # plt.plot(self.x_pred, logistic(self.qm4)*self.qm3, lw=2)
        #
        # plt.subplot(self.nrows, self.ncols, 5)
        # plt.title('activation 1')
        # plt.plot(self.x_pred, logistic(self.qm2), 'g', lw=2)
        # plt.fill_between(self.x_pred, logistic(self.qm2-2*np.sqrt(self.qv2)),
        #                  logistic(self.qm2+2*np.sqrt(self.qv2)), color='g', alpha=0.2)
        #
        # plt.subplot(self.nrows, self.ncols, 6)
        # plt.title('activation 2')
        # plt.plot(self.x_pred, logistic(self.qm4), 'g', lw=2)
        # plt.fill_between(self.x_pred, logistic(self.qm4 - 2*np.sqrt(self.qv4)),
        #                  logistic(self.qm4+2*np.sqrt(self.qv4)), color='g', alpha=0.2)
        #
        # plt.subplot(self.nrows, self.ncols, 7)
        # plt.title('component 1')
        # plt.plot(self.x_pred, self.qm1, color='C0', lw=2)
        # plt.fill_between(self.x_pred, self.qm1-2*np.sqrt(self.qv1), self.qm1+2*np.sqrt(self.qv1),
        #                  color='C0', alpha=0.2)
        #
        # plt.subplot(self.nrows, self.ncols, 8)
        # plt.title('component 2')
        # plt.plot(self.x_pred, self.qm3, color='C0', lw=2)
        # plt.fill_between(self.x_pred, self.qm3-2*np.sqrt(self.qv3), self.qm3+2*np.sqrt(self.qv3),
        #                  color='C0', alpha=0.2)