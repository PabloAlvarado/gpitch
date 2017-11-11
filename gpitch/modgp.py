import numpy as np
import gpflow
from gpflow import settings
from gpflow.minibatch import MinibatchData
import tensorflow as tf
from likelihoods import ModLik
float_type = settings.dtypes.float_type


class ModGP(gpflow.model.Model):
    def __init__(self, x, y, z, kern_com, kern_act, whiten=True, minibatch_size=None):
        gpflow.model.Model.__init__(self)

        if minibatch_size is None:
            minibatch_size = x.shape[0]

        self.num_data = x.shape[0]
        self.x = MinibatchData(x, minibatch_size, np.random.RandomState(0))
        self.y = MinibatchData(y, minibatch_size, np.random.RandomState(0))
        self.z = gpflow.param.Param(z)
        self.kern_com = kern_com
        self.kern_act = kern_act
        self.likelihood = ModLik()
        self.num_inducing = z.shape[0]
        self.whiten = whiten
        # initialize variational parameters
        self.q_mu_com = gpflow.param.Param(np.zeros((self.z.shape[0], 1)))
        self.q_mu_act = gpflow.param.Param(np.zeros((self.z.shape[0], 1)))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)
        self.q_sqrt_com = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt_act = gpflow.param.Param(q_sqrt.copy())

    def build_prior_KL(self):
        if self.whiten:
            KL1 = gpflow.kullback_leiblers.gauss_kl_white(self.q_mu_com,
                                                          self.q_sqrt_com)
            KL2 = gpflow.kullback_leiblers.gauss_kl_white(self.q_mu_act,
                                                          self.q_sqrt_act)
        else:
            K1 = self.kern_com.K(self.z) + \
                 tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            K2 = self.kern_act.K(self.z) + \
                 tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            KL1 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_com,
                                                    self.q_sqrt_com, K1)
            KL2 = gpflow.kullback_leiblers.gauss_kl(self.q_mu_act,
                                                    self.q_sqrt_act, K2)
        return KL1 + KL2

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = gpflow.conditionals.conditional(self.x, self.z,
                                                        self.kern_com, self.q_mu_com,
                                                        q_sqrt=self.q_sqrt_com,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean2, fvar2 = gpflow.conditionals.conditional(self.x, self.z,
                                                        self.kern_act, self.q_mu_act,
                                                        q_sqrt=self.q_sqrt_act,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean = tf.concat([fmean1, fmean2], 1)
        fvar = tf.concat([fvar1, fvar2], 1)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.x)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL


    def predict_all(self, xnew):
        """
        method introduced by Pablo A. Alvarado (11/11/2017)

        This method call all the decorators needed to compute the prediction over the latent
        component and activation. It also reshape the arrays to make easier to plot the intervals of
        confidency.
        """
        mean_f, var_f = self.predict_com(xnew)  # predict component
        mean_g, var_g = self.predict_act(xnew)  # predict activation
        mean_f = mean_f.reshape(-1, )  # reshape arrays in order to be easier plot variances
        var_f = var_f.reshape(-1, )
        mean_g = mean_g.reshape(-1, )
        var_g = var_g.reshape(-1, )
        x_plot = xnew.reshape(-1, ).copy()
        return mean_f, var_f, mean_g, var_g, x_plot

    def fixed_msmkern_params(self, freq=True, var=True):
        """
        method introduced by Pablo A. Alvarado (11/11/2017)

        This methods fixes or unfixes all the params associated to the frequencies and variacnes of
        the matern specrtal mixture kernel.
        """
        Nc = self.kern_com.Nc
        flist = [None]*Nc
        for i in range(Nc):
            flist[i] = 'self.kern_com.frequency_' + str(i + 1) + '.fixed = ' + str(freq)
            exec(flist[i])

        for i in range(Nc):
            flist[i] = 'self.kern_com.variance_' + str(i + 1) + '.fixed = ' + str(var)
            exec(flist[i])



    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_com(self, xnew):
        return gpflow.conditionals.conditional(xnew, self.z, self.kern_com,
                                               self.q_mu_com, q_sqrt=self.q_sqrt_com,
                                               full_cov=False,
                                               whiten=self.whiten)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_act(self, xnew):
        return gpflow.conditionals.conditional(xnew, self.z, self.kern_act,
                                               self.q_mu_act, q_sqrt=self.q_sqrt_act,
                                               full_cov=False,
                                               whiten=self.whiten)





























#
