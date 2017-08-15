import GPflow
import tensorflow as tf
import numpy as np
import itertools


def mvhermgauss(means, covs, H, D):
        """
        Return the evaluation locations, and weights for several multivariate Hermite-Gauss quadrature runs.
        :param means: NxD
        :param covs: NxDxD
        :param H: Number of Gauss-Hermite evaluation points.
        :param D: Number of input dimensions. Needs to be known at call-time.
        :return: eval_locations (H**DxNxD), weights (H**D)
        """
        N = tf.shape(means)[0]
        #gh_x, gh_w = GPflow.kernels.hermgauss(H)
        gh_x, gh_w = GPflow.likelihoods.hermgauss(H)
        xn = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
        wn = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
        cholXcov = tf.cholesky(covs)  # NxDxD
        X = 2.0 ** 0.5 * tf.batch_matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)),
                                         adj_y=True) + tf.expand_dims(means, 2)  # NxDxH**D
        Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, D))  # H**DxNxD
        return Xr, wn * np.pi ** (-D * 0.5)


class ModLik(GPflow.likelihoods.Likelihood):
    def __init__(self):
        GPflow.likelihoods.Likelihood.__init__(self)
        self.noise_var = GPflow.param.Param(1.0)

    def logp(self, F, Y):
        f, g = F[:, 0], F[:, 1]
        y = Y[:, 0]
        sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
        mean = f * sigma_g
        return GPflow.densities.gaussian(y, mean, self.noise_var).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        H, D = 20, 2 # H number of eval points and D dimensions
        Xr, w = mvhermgauss(Fmu, tf.matrix_diag(Fvar), H, D)
        w = tf.reshape(w, [-1, 1])
        f, g = Xr[:, 0], Xr[:, 1]
        y = tf.tile(Y, [H**D, 1])[:, 0]
        sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
        mean = f * sigma_g
        evaluations = GPflow.densities.gaussian(y, mean, self.noise_var)
        evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w), tf.shape(Fmu)[0]])))
        return tf.matmul(evaluations, w)
