import gpflow
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder
from gpflow import settings
import numpy as np

float_type = settings.dtypes.float_type


class SGPRSS(gpflow.sgpr.SGPR):
    """
    Sparse Gaussian process regression for source separation
    """
    def __init__(self, X, Y, kern, Z, mean_function=None, reg=False):

        # if regularization is true
        if reg:
            # introduce vector (ParamList) with the variances of every pitch kernel 
            D = len(kern.kern_list)
            var_list = []
            for i in range(D):
                var_list.append(kern.kern_list[i].variance)
            kern.var_vector = gpflow.param.ParamList(var_list)

        gpflow.sgpr.SGPR.__init__(self, X=X, Y=Y, kern=kern, Z=Z, mean_function=mean_function)
        self.Z = DataHolder(Z, on_shape_change='pass')
        self.reg = reg

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.cast(tf.shape(self.Y)[0], settings.dtypes.float_type)
        output_dim = tf.cast(tf.shape(self.Y)[1], settings.dtypes.float_type)

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(Kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += - output_dim * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.matrix_diag_part(AAT))

        if self.reg:
            # add regularization
            beta = 1000.
            # regularization = -beta * reduce(tf.add, map(tf.abs, self.kern.var_vector))  # L-1 norm
            regularization = -beta * tf.reduce_sum(reduce(tf.add, map(tf.abs, self.kern.var_vector)))
            return bound + regularization
        else:
            return bound



    def build_predict_source(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(source* | Y )

        where source* are points on the source at Xnew, Y are noisy observations of the mixture at X.

        """
        mean = []
        var = []
        nsources = len(self.kern.kern_list)

        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))

        for i in range(nsources):
            Kx = self.kern.kern_list[i].K(self.X, Xnew)
            A = tf.matrix_triangular_solve(L, Kx, lower=True)
            smean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
            if full_cov:
                svar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
                shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
                svar = tf.tile(tf.expand_dims(svar, 2), shape)
            else:
                svar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
                svar = tf.tile(tf.reshape(svar, (-1, 1)), [1, tf.shape(self.Y)[1]])

            mean.append(smean)
            var.append(svar)
        return mean, var

    @AutoFlow((float_type, [None, None]))
    def predict_s(self, Xnew):
        """
        Compute the mean and variance of the sources
        at the points `Xnew`.
        """
        return self.build_predict_source(Xnew)
