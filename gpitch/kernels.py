from __future__ import print_function, absolute_import
from functools import reduce
import itertools
import warnings
import tensorflow as tf
import numpy as np
from gpflow.param import Param, Parameterized, AutoFlow
from gpflow import transforms
from gpflow._settings import settings
float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Matern12Cosine(gpflow.kernels.Kern):
    """
    Matern 1/2 times Cosine kernel
    """

    def __init__(self, input_dim, period=1.0, variance=1.0,
                 lengthscales=1.0, active_dims=None):
        # No ARD support for lengthscale or period yet
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(variance, transforms.positive)
        self.lengthscales = Param(lengthscales, transforms.positive)
        self.ARD = False
        self.period = Param(period, transforms.positive)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = tf.sqrt(tf.square(f - f2))
        r1 = tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi*r / self.period, 2)

        return self.variance * tf.exp(-r1) * tf.cos(r2)


class Inharmonic(kernels.Kern):
    '''Inharmonic kernel. Useful for modelling piano sounds.'''
    def __init__(self, input_dim, lengthscales, variances, beta, f0):
        kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.Nc = lengthscales.size
        self.ARD = False
        #  generate a param object for each lengthscale and variance.
        #  lengthscales and variances must be (Nc,) arrays.
        for i in range(self.Nc):
            setattr(self, 'lengthscale_' + str(i+1), param.Param(lengthscales[i], transforms.positive) )
            setattr(self, 'variance_' + str(i+1), param.Param(variances[i], transforms.positive) )
        self.beta = param.Param(beta, transforms.positive)
        self.f0 = param.Param(f0, transforms.positive)


    def K(self, arg):
        pass

    def Kdiag(self, arg):
        pass
























#
