import numpy as np
import GPflow
from GPflow import settings
import tensorflow as tf
import modulating_likelihood_old
reload(modulating_likelihood_old)
from modulating_likelihood_old import ModLik


class ModGP(GPflow.model.Model):
    def __init__(self, X, Y, kern1, kern2, Z):
        GPflow.model.Model.__init__(self)
        self.X, self.Y, self.kern1, self.kern2 = X, Y, kern1, kern2
        self.likelihood = ModLik()
        self.Z = Z
        self.num_inducing = Z.shape[0]
        self.num_data = X.shape[0]

        self.q_mu1 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        self.q_mu2 = GPflow.param.Param(np.zeros((self.Z.shape[0], 1)))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)

        self.q_sqrt1 = GPflow.param.Param(q_sqrt.copy())
        self.q_sqrt2 = GPflow.param.Param(q_sqrt.copy())

    def build_prior_KL(self):
        K1 = self.kern1.K(self.Z) + np.eye(self.num_inducing) * settings.numerics.jitter_level
        KL1 = GPflow.kullback_leiblers.gauss_kl(self.q_mu1, self.q_sqrt1, K1)
        K2 = self.kern2.K(self.Z) + np.eye(self.num_inducing) * settings.numerics.jitter_level
        KL2 = GPflow.kullback_leiblers.gauss_kl(self.q_mu2, self.q_sqrt2, K2)
        return KL1 + KL2

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = GPflow.conditionals.conditional(self.X, self.Z, self.kern1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1, full_cov=False, whiten=False)

        fmean2, fvar2 = GPflow.conditionals.conditional(self.X, self.Z, self.kern2, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2, full_cov=False, whiten=False)

        fmean, fvar = tf.concat(1, [fmean1, fmean2]), tf.concat(1, [fvar1, fvar2])

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern1, self.q_mu1,
                                               q_sqrt=self.q_sqrt1, full_cov=False, whiten=False)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern2, self.q_mu2,
                                               q_sqrt=self.q_sqrt2, full_cov=False, whiten=False)
