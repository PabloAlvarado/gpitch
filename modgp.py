import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.minibatch import MinibatchData
from likelihoods import ModLik


float_type = settings.dtypes.float_type


class ModGP(gpflow.model.Model):
    def __init__(self, x, y, z, kern_com, kern_act, transfunc, whiten=True, minibatch_size=None):
        """
        Constructor.
        :param x:
        :param y:
        :param z:
        :param kern_com:
        :param kern_act:
        :param transform:
        :param whiten:
        :param minibatch_size:
        """

        gpflow.model.Model.__init__(self)

        if minibatch_size is None:
            minibatch_size = x.shape[0]

        self.minibatch_size = minibatch_size
        self.num_data = x.shape[0]
        self.x = MinibatchData(x, minibatch_size, np.random.RandomState(0))
        self.y = MinibatchData(y, minibatch_size, np.random.RandomState(0))
        self.z = gpflow.param.Param(z)
        self.kern_com = kern_com
        self.kern_act = kern_act
        self.likelihood = ModLik(transfunc)
        self.whiten = whiten
        self.q_mu_com = gpflow.param.Param(np.zeros((self.z.shape[0], 1)))  # initialize variational parameters
        self.q_mu_act = gpflow.param.Param(np.zeros((self.z.shape[0], 1)))
        self.num_inducing = z.shape[0]
        q_sqrt = np.array([np.eye(self.num_inducing) for _ in range(1)]).swapaxes(0, 2)
        self.q_sqrt_com = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt_act = gpflow.param.Param(q_sqrt.copy())
        self.logf = []  # used for store values when using svi

    def build_prior_kl(self):
        """
        compute KL divergences.
        :return:
        """
        if self.whiten:
            kl1 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_com, self.q_sqrt_com)
            kl2 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_act, self.q_sqrt_act)
        else:
            k1 = self.kern_com.K(self.z) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            k2 = self.kern_act.K(self.z) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            kl1 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_com, self.q_sqrt_com, k1)
            kl2 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_act, self.q_sqrt_act, k2)
        return kl1 + kl2

    def build_likelihood(self):
        """
        Compute the objective function
        :return:
        """
        kl = self.build_prior_kl()  # Get prior kl.

        # Get conditionals
        fmean1, fvar1 = gpflow.conditionals.conditional(self.x, self.z, self.kern_com, self.q_mu_com,
                                                        q_sqrt=self.q_sqrt_com, full_cov=False, whiten=self.whiten)
        fmean2, fvar2 = gpflow.conditionals.conditional(self.x, self.z, self.kern_act, self.q_mu_act,
                                                        q_sqrt=self.q_sqrt_act, full_cov=False,  whiten=self.whiten)
        fmean = tf.concat([fmean1, fmean2], 1)
        fvar = tf.concat([fvar1, fvar2], 1)

        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.y)  # Get variational expectations

        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.x)[0], settings.dtypes.float_type)  # re-scale for minibatch size
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_all(self, xnew):
        """
        method introduced by Pablo A. Alvarado (11/11/2017)

        This method call all the decorators needed to compute the prediction over the latent
        component and activation. It also reshape the arrays to make easier to plot the intervals of
        confidency.

        INPUT
        xnew : list of array to use for prediction
        """
        l_com_mean = []  # list to storage predictions
        l_com_var = []
        l_act_mean = []
        l_act_var = []
        for i in range(len(xnew)):
            # print('predicting window ' + str(i + 1) + ' of ' + str(len(xnew)))
            mean_f, var_f = self.predict_com(xnew[i])  # predict component
            mean_g, var_g = self.predict_act(xnew[i])  # predict activation
            l_com_mean.append(mean_f)
            l_com_var.append(var_f)
            l_act_mean.append(mean_g)
            l_act_var.append(var_g)

        mean_f = np.asarray(l_com_mean).reshape(-1,)
        mean_g = np.asarray(l_act_mean).reshape(-1,)
        var_f = np.asarray(l_com_var).reshape(-1,)
        var_g = np.asarray(l_act_var).reshape(-1,)
        x_plot = np.asarray(xnew).reshape(-1,)
        return mean_f, var_f, mean_g, var_g, x_plot

    def fixed_msmkern_params(self, freq=True, var=True):
        """
        method introduced by Pablo A. Alvarado (11/11/2017)

        This methods fixes or unfixes all the params associated to the frequencies and variances of
        the matern specrtal mixture kernel.
        """
        nc = self.kern_com.Nc
        flist = [None]*nc
        for i in range(nc):
            flist[i] = 'self.kern_com.frequency_' + str(i + 1) + '.fixed = ' + str(freq)
            exec(flist[i])

        for i in range(nc):
            flist[i] = 'self.kern_com.variance_' + str(i + 1) + '.fixed = ' + str(var)
            exec(flist[i])

    def optimize_svi(self, maxiter, learning_rate=0.001):
        """
        method introduced by Pablo A. Alvarado (20/11/2017)
        This method uses stochastic variational inference for maximizing the ELBO.
        """
        def logger(x):
            if (logger.i % 10) == 0:
                self.logf.append(self._objective(x)[0])
            logger.i += 1
        logger.i = 1
        self.x.minibatch_size = self.minibatch_size
        self.y.minibatch_size = self.minibatch_size
        method = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimize(method=method, maxiter=maxiter, callback=logger)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_com(self, xnew):
        return gpflow.conditionals.conditional(xnew, self.z, self.kern_com,
                                               self.q_mu_com, q_sqrt=self.q_sqrt_com,
                                               full_cov=False, whiten=self.whiten)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_act(self, xnew):
        return gpflow.conditionals.conditional(xnew, self.z, self.kern_act,
                                               self.q_mu_act, q_sqrt=self.q_sqrt_act,
                                               full_cov=False, whiten=self.whiten)
