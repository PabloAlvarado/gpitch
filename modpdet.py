import numpy as np
import gpflow
import modgp
import amtgp


class ModPDet:
    """
    Gaussian process pitch detection using modulated GP
    """
    def __init__(self, x, y, ker_com, ker_act, ws, dec, Nw=None, bounded=False, whiten=True):
        self.x, self.y = x, y
        self.ws = ws # sample rate, and window size (samples)
        self.dec, self.N = dec, x.size
        if Nw == None: self.Nw = self.N/self.ws  # number of windows
        else: self.Nw = 1
        self.bounded = bounded #  bound inducing variables for f_m to be (-1, 1)
        self.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)] # split data into windows
        self.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)]
        self.z = np.vstack(( self.x_l[0][::dec].copy(), self.x_l[0][-1].copy() ))

        # initialize modulated GP model and unfix component parameters
        self.model = modgp.ModGP(x=self.x_l[0].copy(), y=self.y_l[0].copy(), kern_com=ker_com, kern_act= ker_act, z=self.z, whiten=whiten)


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
            self.model.x = self.x_l[i].copy()
            self.model.y = self.y_l[i].copy()
            self.model.z = np.vstack(( self.x_l[i][::self.dec].copy(), self.x_l[i][-1].copy() ))
            if init_zero:
                self.model.q_mu1 = np.zeros(self.model.z.shape)
                self.model.q_mu2 = np.zeros(self.model.z.shape)
                self.model.q_sqrt1 = np.expand_dims(np.eye(self.model.z.size), 2)
                self.model.q_sqrt2 = np.expand_dims(np.eye(self.model.z.size), 2)

            self.model.optimize(disp=disp, maxiter=maxiter)
            self.qm1_l[i], self.qv1_l[i] = self.model.predict_com(self.x_l[i])
            self.qm2_l[i], self.qv2_l[i] = self.model.predict_act(self.x_l[i])
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
            self.model.kern_com.lengthscales = 0.1*np.random.rand()
            self.model.kern_act.lengthscales = 1.*np.random.rand()
            self.model.kern_act.variance = 15.*np.random.rand()
            self.model.likelihood.variance = 0.1*np.random.rand()

            init_hyper[0][r] = self.model.kern_act.lengthscales.value
            init_hyper[1][r] = self.model.kern_act.variance.value
            init_hyper[2][r] = self.model.likelihood.variance.value
            init_hyper[3][r] = self.model.kern_com.lengthscales.value

            self.optimize_windowed(disp=0, maxiter=maxiter)
            learnt_hyper[0][r] = self.model.kern_act.lengthscales.value
            learnt_hyper[1][r] = self.model.kern_act.variance.value
            learnt_hyper[2][r] = self.model.likelihood.variance.value
            learnt_hyper[3][r] = self.model.kern_com.lengthscales.value
            mse[r] = (1./self.N)*np.sum((self.y_pred - amtgp.logistic(self.qm2)*self.qm1)**2)
            print('| len: %4.4f, %4.4f | sig: %4.4f, %4.4f | noise_var: %4.4f, %4.4f | l_com: %4.4f, %4.4f |' % (init_hyper[0][r], learnt_hyper[0][r],
                                                                                                                 init_hyper[1][r], learnt_hyper[1][r],
                                                                                                                 init_hyper[2][r], learnt_hyper[2][r],
                                                                                                                 init_hyper[3][r], learnt_hyper[3][r]) )
        return init_hyper, learnt_hyper, mse


    def optimize(self, maxiter=10, restarts=1):

        self.model.likelihood.noise_var.transform = gpflow.transforms.Logistic(a=0., b=1e-1)  # bound noise variance
        self.model.likelihood.noise_var.fixed = False
        self.model.kern2.fixed = False  # unfix params of kernel for activation
        self.model.kern1.fixed = True  # set component kernel to only variances fixed
        self.model.kern1.lengthscales.fixed = False
        self.model.kern1.frequency_1.fixed = False
        self.model.kern1.frequency_2.fixed = False
        self.model.kern1.frequency_3.fixed = False
        self.model.kern1.frequency_4.fixed = False
        self.model.kern1.frequency_5.fixed = False
        self.model.kern1.frequency_6.fixed = False
        self.model.kern1.frequency_7.fixed = False
        self.model.kern1.frequency_8.fixed = False
        self.model.kern1.frequency_9.fixed = False
        self.model.kern1.frequency_10.fixed = False
        self.model.kern1.frequency_11.fixed = False
        self.model.kern1.frequency_12.fixed = False
        self.model.kern1.frequency_13.fixed = False
        self.model.kern1.frequency_14.fixed = False
        self.model.kern1.frequency_15.fixed = False

        init_hyper, learnt_hyper, mse = self.optimize_restart(maxiter=maxiter, restarts=restarts)

        self.model.kern2.lengthscales = learnt_hyper[0].mean().copy()
        self.model.kern2.variance = learnt_hyper[1].mean().copy()
        self.model.likelihood.noise_var = learnt_hyper[2].mean().copy()
        self.model.kern1.lengthscales = learnt_hyper[3].mean().copy()
        self.model.optimize(disp=0, maxiter=maxiter)

        self.model.kern1.fixed = True
        self.model.kern2.fixed = True
        self.model.likelihood.noise_var.fixed = True

        self.model.kern1.variance_1.fixed = False
        self.model.kern1.variance_2.fixed = False
        self.model.kern1.variance_3.fixed = False
        self.model.kern1.variance_4.fixed = False
        self.model.kern1.variance_5.fixed = False
        self.model.kern1.variance_6.fixed = False
        self.model.kern1.variance_7.fixed = False
        self.model.kern1.variance_8.fixed = False
        self.model.kern1.variance_9.fixed = False
        self.model.kern1.variance_10.fixed = False
        self.model.kern1.variance_11.fixed = False
        self.model.kern1.variance_12.fixed = False
        self.model.kern1.variance_13.fixed = False
        self.model.kern1.variance_14.fixed = False
        self.model.kern1.variance_15.fixed = False

        self.model.optimize(disp=0, maxiter=10)

    def predict_all(self, x):
        mean_f, var_f = self.model.predict_f(x)  # predict component
        mean_g, var_g = self.model.predict_g(x)  # predict activation
        mean_f = mean_f.reshape(-1, )  # reshape arrays in order to be easier plot variances
        var_f = var_f.reshape(-1, )
        mean_g = mean_g.reshape(-1, )
        var_g = var_g.reshape(-1, )
        x_plot = x.reshape(-1, ).copy()
        return mean_f, var_f, mean_g, var_g, x_plot


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
