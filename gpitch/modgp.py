import numpy as np
import gpflow
from gpflow import settings
from gpflow.minibatch import MinibatchData
import tensorflow as tf
from likelihoods import ModLik
float_type = settings.dtypes.float_type


class ModGP(gpflow.model.Model):
    def __init__(self, X, Y, kern1, kern2, Z, whiten=False, minibatch_size=None):
        gpflow.model.Model.__init__(self)
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        #self.X = X
        #self.Y = Y
        self.X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        self.Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))
        self.Z = gpflow.param.DataHolder(Z, on_shape_change='pass')

        self.kern1, self.kern2 = kern1, kern2
        self.likelihood = ModLik()
        self.num_inducing = Z.shape[0]
        self.whiten = whiten

        # initialize variational parameters
        self.q_mu1 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu2 = gpflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)
        self.q_sqrt1 = gpflow.param.Param(q_sqrt.copy())
        self.q_sqrt2 = gpflow.param.Param(q_sqrt.copy())

    def build_prior_KL(self):
        if self.whiten:
            KL1 = gpflow.kullback_leiblers.gauss_kl_white(self.q_mu1,
                                                          self.q_sqrt1)
            KL2 = gpflow.kullback_leiblers.gauss_kl_white(self.q_mu2,
                                                          self.q_sqrt2)
        else:
            K1 = self.kern1.K(self.Z) + \
                 tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            K2 = self.kern2.K(self.Z) + \
                 tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            KL1 = gpflow.kullback_leiblers.gauss_kl(self.q_mu1,
                                                    self.q_sqrt1, K1)
            KL2 = gpflow.kullback_leiblers.gauss_kl(self.q_mu2,
                                                    self.q_sqrt2, K2)
        return KL1 + KL2

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = gpflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean2, fvar2 = gpflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern2, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean = tf.concat([fmean1, fmean2], 1)
        fvar = tf.concat([fvar1, fvar2], 1)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern1,
                                               self.q_mu1, q_sqrt=self.q_sqrt1,
                                               full_cov=False,
                                               whiten=self.whiten)

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g(self, Xnew):
        return gpflow.conditionals.conditional(Xnew, self.Z, self.kern2,
                                               self.q_mu2, q_sqrt=self.q_sqrt2,
                                               full_cov=False,
                                               whiten=self.whiten)






























#
