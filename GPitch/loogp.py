import numpy as np
import GPflow
from GPflow import settings
from GPflow.kullback_leiblers import gauss_kl, gauss_kl_white
from GPflow.minibatch import MinibatchData
import tensorflow as tf
import loo_likelihood
reload(loo_likelihood)
from loo_likelihood import LooLik
jitter = settings.numerics.jitter_level


class LooGP(GPflow.model.Model):
    def __init__(self, X, Y, kf, kg, Z, whiten=True, minibatch_size=None,
                 old_version=False):
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

        self.Z = GPflow.param.DataHolder(Z, on_shape_change='pass')
        #self.Z = Z
        #self.Z = GPflow.param.Param(Z)

        self.kern_f1, self.kern_f2  = kf[0], kf[1]
        self.kern_g1, self.kern_g2 = kg[0], kg[1]
        self.likelihood = LooLik(version=old_version)
        self.num_inducing = Z.shape[0]
        self.whiten = whiten

        # initialize variational parameters
        self.q_mu1, self.q_mu2, self.q_mu3, self.q_mu4 = \
        [GPflow.param.Param(np.zeros((self.Z.shape[0], 1))) for _ in range(4)]

        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(1)]).swapaxes(0, 2)

        self.q_sqrt1, self.q_sqrt2, self.q_sqrt3, self.q_sqrt4 = \
        [GPflow.param.Param(q_sqrt.copy()) for _ in range(4)]

    def build_prior_KL(self):
        if self.whiten:
            KL1 = gauss_kl_white(self.q_mu1, self.q_sqrt1)
            KL2 = gauss_kl_white(self.q_mu2, self.q_sqrt2)
            KL3 = gauss_kl_white(self.q_mu3, self.q_sqrt3)
            KL4 = gauss_kl_white(self.q_mu4, self.q_sqrt4)
        else:
            K1 = self.kern_f1.K(self.Z) + np.eye(self.num_inducing) * jitter
            K2 = self.kern_g1.K(self.Z) + np.eye(self.num_inducing) * jitter
            K3 = self.kern_f2.K(self.Z) + np.eye(self.num_inducing) * jitter
            K4 = self.kern_g2.K(self.Z) + np.eye(self.num_inducing) * jitter
            KL1 = gauss_kl(self.q_mu1, self.q_sqrt1, K1)
            KL2 = gauss_kl(self.q_mu2, self.q_sqrt2, K2)
            KL3 = gauss_kl(self.q_mu3, self.q_sqrt3, K3)
            KL4 = gauss_kl(self.q_mu4, self.q_sqrt4, K4)
        return KL1 + KL2 + KL3 + KL4

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern_f1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean2, fvar2 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern_g1, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean3, fvar3 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern_f2, self.q_mu3,
                                                        q_sqrt=self.q_sqrt3,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean4, fvar4 = GPflow.conditionals.conditional(self.X, self.Z,
                                                        self.kern_g2, self.q_mu4,
                                                        q_sqrt=self.q_sqrt4,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        fmean = tf.concat(1, [fmean1, fmean2, fmean3, fmean4])
        fvar = tf.concat(1, [fvar1, fvar2, fvar3, fvar4])

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f1(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern_f1,
                                               self.q_mu1, q_sqrt=self.q_sqrt1,
                                               full_cov=False,
                                               whiten=self.whiten)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g1(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern_g1,
                                               self.q_mu2, q_sqrt=self.q_sqrt2,
                                               full_cov=False,
                                               whiten=self.whiten)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_f2(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern_f2,
                                               self.q_mu3, q_sqrt=self.q_sqrt3,
                                               full_cov=False,
                                               whiten=self.whiten)

    @GPflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_g2(self, Xnew):
        return GPflow.conditionals.conditional(Xnew, self.Z, self.kern_g2,
                                               self.q_mu4, q_sqrt=self.q_sqrt4,
                                               full_cov=False,
                                               whiten=self.whiten)





























#
