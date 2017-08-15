import numpy as np
import GPflow
from GPflow import settings
from GPflow.minibatch import MinibatchData
import tensorflow as tf
import loo_likelihood
reload(loo_likelihood)
from loo_likelihood import LooLik


class LooGP(GPflow.model.Model):
    def __init__(self, X, Y, kf, kg, Z, whiten=True, minibatch_size=None):
        '''Leave One Out (LOO) model.
        INPUTS:
        kf : list of kernels for each latent quasi-periodic function
        kg : list of kernels for each latent envelope function
        '''
        GPflow.model.Model.__init__(self)

        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]

        self.X = X
        self.Y = Y
        self.X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        self.Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))
        #self.X = GPflow.param.DataHolder(X, on_shape_change='pass')
        #self.Y = GPflow.param.DataHolder(Y, on_shape_change='pass')

        #self.Z = Z
        self.Z = GPflow.param.DataHolder(Z, on_shape_change='pass')
        #self.Z = GPflow.param.Param(Z)

        self.kern1, self.kern2 = kf[0], kf[1]
        self.kern3, self.kern4 = kg[2], kg[3]
        self.likelihood = LooLik()
        self.num_inducing = Z.shape[0]
        self.whiten = whiten

        # initialize variational parameters
        self.q_mu1 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu2 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu3 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu4 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))

        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)
        self.q_sqrt1 = GPflow.param.Param(q_sqrt.copy())
        self.q_sqrt2 = GPflow.param.Param(q_sqrt.copy())
        self.q_sqrt3 = GPflow.param.Param(q_sqrt.copy())
        self.q_sqrt4 = GPflow.param.Param(q_sqrt.copy())

    def build_prior_KL(self):
        if self.whiten:
            KL1 = GPflow.kullback_leiblers.gauss_kl_white(self.q_mu1,
                                                          self.q_sqrt1)
            KL2 = GPflow.kullback_leiblers.gauss_kl_white(self.q_mu2,
                                                          self.q_sqrt2)
        else:
            K1 = self.kern1.K(self.Z) + \
                 np.eye(self.num_inducing) * settings.numerics.jitter_level
            K2 = self.kern2.K(self.Z) + \
                 np.eye(self.num_inducing) * settings.numerics.jitter_level
            KL1 = GPflow.kullback_leiblers.gauss_kl(self.q_mu1,
                                                    self.q_sqrt1, K1)
            KL2 = GPflow.kullback_leiblers.gauss_kl(self.q_mu2,
                                                    self.q_sqrt2, K2)
        return KL1 + KL2

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean2, fvar2 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern2, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean = tf.concat(1, [fmean1, fmean2])
        fvar = tf.concat(1, [fvar1, fvar2])

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern1,
                                               self.q_mu1, q_sqrt=self.q_sqrt1,
                                               full_cov=False,
                                               whiten=self.whiten)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern2,
                                               self.q_mu2, q_sqrt=self.q_sqrt2,
                                               full_cov=False,
                                               whiten=self.whiten)






























#