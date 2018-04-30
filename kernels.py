import numpy as np
import tensorflow as tf
import gpflow
from scipy.fftpack import fft, ifft, ifftshift
from gpflow.param import Param, transforms
from gpflow._settings import settings


float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Matern32sm(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, numc, lengthscales=None, variances=None, frequencies=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        self.numc = numc

        if lengthscales == None:
            lengthscales = 0.05
            variances = 0.125*np.ones((numc, 1))
            frequencies = 1.*np.arange(1, numc+1)

        self.lengthscales = Param(lengthscales, transforms.Logistic(0., 10.) )
        for i in range(self.numc): # generate a param object for each  var, and freq, they must be (numc,) arrays.
            setattr(self, 'variance_' + str(i+1), Param(variances[i], transforms.Logistic(0., 0.25) ) )
            setattr(self, 'frequency_' + str(i+1), Param(frequencies[i], transforms.positive ) )

        for i in range(self.numc):
            exec('self.variance_' + str(i + 1) + '.fixed = ' + str(True))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = np.sqrt(3.)*tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency_1 * r , 2)
        k = self.variance_1 * (1. + r1) * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.numc + 1):
            r2 = tf.reduce_sum(2.*np.pi * getattr(self, 'frequency_' + str(i)) * r , 2)
            k += getattr(self, 'variance_' + str(i)) * (1. + r1) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.numc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var


















#
