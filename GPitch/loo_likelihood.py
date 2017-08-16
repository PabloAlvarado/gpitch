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

# Pablo Alvarado implementation
def hermgauss1d(mean_g, var_g, H):
    #H = 20  # get eval points and weights
    gh_x, gh_w = GPflow.quadrature.hermgauss(H)
    gh_x = gh_x.reshape(1, -1)
    gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)

    shape = tf.shape(mean_g)  # get  output shape
    X = gh_x * tf.sqrt(2.*var_g) + mean_g  # transformed evaluation points
    evaluations = 1. / (1. + tf.exp(-X))  # sigmoid function
    E1 = tf.reshape(tf.matmul(evaluations, gh_w), shape)  # compute expect
    E2 = tf.reshape(tf.matmul(evaluations**2, gh_w), shape)
    return E1, E2

class LooLik(GPflow.likelihoods.Likelihood):
    '''Leave One Out likelihood'''
    def __init__(self):
        GPflow.likelihoods.Likelihood.__init__(self)
        self.noise_var = GPflow.param.Param(1.0)

    def logp(self, F, Y):
        f1, g1 = F[:, 0], F[:, 1]
        f2, g2 = F[:, 2], F[:, 3]
        y = Y[:, 0]
        sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
        sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
        mean = sigma_g1 * f1 + sigma_g2 * f2
        return GPflow.densities.gaussian(y, mean, self.noise_var).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        old_version = False
        if old_version:
            H = 10 # number of Gauss-Hermite evaluation points. (reduced  to 5)
            D = 4  # Number of input dimensions (increased from 2 to 4)
            Xr, w = mvhermgauss(Fmu, tf.matrix_diag(Fvar), H, D)
            w = tf.reshape(w, [-1, 1])
            f1, g1 = Xr[:, 0], Xr[:, 1]
            f2, g2 = Xr[:, 2], Xr[:, 3]
            y = tf.tile(Y, [H**D, 1])[:, 0]
            sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
            sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
            mean =  sigma_g1 * f1 + sigma_g2 * f2
            evaluations = GPflow.densities.gaussian(y, mean, self.noise_var)
            evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w),
                                                                tf.shape(Fmu)[0]])))
            return tf.matmul(evaluations, w)

        else:
            # variational expectations function, Pablo Alvarado implementation
            mean_f1 = Fmu[:, 0]  # get mean and var of each q dist, and reshape
            mean_g1 = Fmu[:, 1]
            var_f1 = Fvar[:, 0]
            var_g1 = Fvar[:, 1]

            mean_f2 = Fmu[:, 2]  # get mean and var of each q dist, and reshape
            mean_g2 = Fmu[:, 3]
            var_f2 = Fvar[:, 2]
            var_g2 = Fvar[:, 3]

            mean_f1, mean_g1, var_f1, var_g1 = [tf.reshape(e, [-1, 1]) for e in
                                               (mean_f1, mean_g1, var_f1, var_g1)]

            mean_f2, mean_g2, var_f2, var_g2 = [tf.reshape(e, [-1, 1]) for e in
                                               (mean_f2, mean_g2, var_f2, var_g2)]
            H = 10
            # calculate required quadratures
            E1, E2 = hermgauss1d(mean_g1, var_g1, H)
            E3, E4 = hermgauss1d(mean_g2, var_g2, H)

            # compute log-lik expectations under variational distribution
            var_exp = -0.5*((1./self.noise_var)*(Y**2 -
                       2.*Y*(mean_f1*E1 + mean_f2*E3) +
                      (var_f1 + mean_f1**2)*E2 +
                      2.* mean_f1*E1 * mean_f2*E3 +
                      (var_f2 + mean_f2**2)*E4) +
                      np.log(2.*np.pi) +
                      tf.log(self.noise_var))
            return var_exp






























#
