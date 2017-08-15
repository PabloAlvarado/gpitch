import GPflow
import tensorflow as tf
import numpy as np
import itertools


def mvhermgauss(means, covs, H, D):
        """
        Return the evaluation locations, and weights for several multivariate
        Hermite-Gauss quadrature runs.
        :param means: NxD
        :param covs: NxDxD
        :param H: Number of Gauss-Hermite evaluation points.
        :param D: Number of input dimensions. Needs to be known at call-time.
        :return: eval_locations (H**DxNxD), weights (H**D)
        """
        N = tf.shape(means)[0]
        gh_x, gh_w = GPflow.likelihoods.hermgauss(H)
        xn = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
        wn = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
        cholXcov = tf.cholesky(covs)  # NxDxD
        X = 2.0 ** 0.5 * tf.batch_matmul(cholXcov, tf.tile(xn[None, :, :],
                                         (N, 1, 1)), adj_y=True) + \
                                        tf.expand_dims(means, 2)  # NxDxH**D
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

    # variational expectations function, Pablo Alvarado implementation
    def variational_expectations(self, Fmu, Fvar, Y):
        H = 20  # get eval points and weights
        gh_x, gh_w = GPflow.quadrature.hermgauss(H)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)

        mean_f = Fmu[:, 0]  # get mean and var of each q distribution, and reshape
        mean_g = Fmu[:, 1]
        var_f = Fvar[:, 0]
        var_g = Fvar[:, 1]
        mean_f, mean_g, var_f, var_g = [tf.reshape(e, [-1, 1]) for e in (mean_f,
                                        mean_g, var_f, var_g)]
        shape = tf.shape(mean_g)  # get  output shape
        X = gh_x * tf.sqrt(2.*var_g) + mean_g  # transformed evaluation points
        evaluations = 1. / (1. + tf.exp(-X))  # sigmoid function
        E1 = tf.reshape(tf.matmul(evaluations, gh_w), shape)  # compute expectations
        E2 = tf.reshape(tf.matmul(evaluations**2, gh_w), shape)

        # compute log-lik expectations under variational distribution
        var_exp = -0.5*((1./self.noise_var)*(Y**2 - 2.*Y*mean_f*E1 +
                  (var_f + mean_f**2)*E2) + np.log(2.*np.pi) +
                  tf.log(self.noise_var))
        return var_exp

    # # variational expectations function, GPflow modulated_GP version
    # def variational_expectations(self, Fmu, Fvar, Y):
    #     H = 20
    #     D = 2
    #     Fvar_matrix_diag = tf.matrix_diag(Fvar)
    #     Xr, w = mvhermgauss(Fmu, Fvar_matrix_diag, H, D)
    #     w = tf.reshape(w, [-1, 1])
    #     f, g = Xr[:, 0], Xr[:, 1]
    #     y = tf.tile(Y, [H**D, 1])[:, 0]
    #     sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
    #     mean = f * sigma_g
    #     evaluations = GPflow.densities.gaussian(y, mean, self.noise_var)
    #     evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w),
    #                                tf.shape(Fmu)[0]])))
    #     n_var_exp = tf.matmul(evaluations, w)
    #     return n_var_exp






























#
