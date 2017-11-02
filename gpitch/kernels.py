import numpy as np
import tensorflow as tf
import gpflow
from scipy.fftpack import fft, ifft, ifftshift
from gpflow.param import Param, transforms
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


class Inharmonic(gpflow.kernels.Kern):
    '''
    Inharmonic kernel. Useful for modelling piano sounds.
    '''

    def __init__(self, input_dim, lengthscales, variances, beta, f0):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.Nc = lengthscales.size
        self.ARD = False
        #  generate a param object for each lengthscale and variance.
        #  lengthscales and variances must be (Nc,) arrays.
        for i in range(self.Nc):
            setattr(self, 'lengthscale_' + str(i+1), Param(lengthscales[i], transforms.positive) )
            setattr(self, 'variance_' + str(i+1), Param(variances[i], transforms.positive) )
        self.beta = Param(beta, transforms.positive) #  inharmonicity param
        self.f0 = Param(f0, transforms.positive) #  natural frequency Hz


    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2))

        r1 = tf.reduce_sum(r / self.lengthscale_1, 2)
        r2 = tf.reduce_sum(2.*np.pi*self.f0*tf.sqrt(1. + self.beta)*r , 2)
        k = self.variance_1 * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.Nc + 1):
            r1 = tf.reduce_sum(r / getattr(self, 'lengthscale_' + str(i)), 2)
            r2 = tf.reduce_sum(2.*np.pi*i*self.f0*tf.sqrt(1. + self.beta*(i**2))*r , 2)
            k += getattr(self, 'variance_' + str(i)) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.Nc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var


class MaternSpecMix(gpflow.kernels.Kern):
    '''
    Matern spectral mixture kernel.
    '''

    def __init__(self, input_dim, lengthscales=None, variances=None, frequencies=None, ideal_f0=None, Nc=1):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        if ideal_f0 == None:
            self.Nc = lengthscales.size
            for i in range(self.Nc): # generate a param object for each len, var, and freq, they must be (Nc,) arrays.
                setattr(self, 'lengthscale_' + str(i+1), Param(lengthscales[i], transforms.positive ) )
                setattr(self, 'variance_' + str(i+1), Param(variances[i],transforms.positive ) )
                setattr(self, 'frequency_' + str(i+1), Param(frequencies[i], transforms.positive ) )

        else:
            self.Nc = Nc
            for i in range(Nc): # generate a param object for each len, var, and freq, they must be (Nc,) arrays.
                setattr(self, 'lengthscale_' + str(i+1), Param(0.25, transforms.Logistic(0., .5) ) )
                setattr(self, 'variance_' + str(i+1), Param(0.125, transforms.Logistic(0., 0.25) ) )
                setattr(self, 'frequency_' + str(i+1), Param((i+1)*ideal_f0, transforms.positive ) )

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = tf.reduce_sum(r / self.lengthscale_1, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency_1 * r , 2)
        k = self.variance_1 * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.Nc + 1):
            r1 = tf.reduce_sum(r / getattr(self, 'lengthscale_' + str(i)), 2)
            r2 = tf.reduce_sum(2.*np.pi * getattr(self, 'frequency_' + str(i)) * r , 2)
            k += getattr(self, 'variance_' + str(i)) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.Nc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var



class MaternSpecMixSL(gpflow.kernels.Kern):
    '''
    Matern spectral mixture kernel with single lengthscale.
    '''
    def __init__(self, Nc, input_dim=1, lengthscales=None, variances=None, frequencies=None, ideal_f0=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        self.ideal_f0 = ideal_f0
        if ideal_f0 == None:
            self.lengthscales = Param(lengthscales, transforms.Logistic(0., 1.0) )
            self.Nc = variances.size
            for i in range(self.Nc): # generate a param object for each  var, and freq, they must be (Nc,) arrays.
                setattr(self, 'variance_' + str(i+1), Param(variances[i], transforms.Logistic(0., 0.25) ) )
                setattr(self, 'frequency_' + str(i+1), Param(frequencies[i], transforms.positive ) )

        else:
            self.lengthscales = Param(0.05, transforms.Logistic(0., 0.1) )
            self.Nc = Nc
            for i in range(Nc): # generate a param object for each len, var, and freq, they must be (Nc,) arrays.
                setattr(self, 'variance_' + str(i+1), Param(0.125, transforms.Logistic(0., 0.25) ) )
                setattr(self, 'frequency_' + str(i+1), Param((i+1)*ideal_f0, transforms.positive ) )

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency_1 * r , 2)
        k = self.variance_1 * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.Nc + 1):
            r1 = tf.reduce_sum(r / self.lengthscales, 2)
            r2 = tf.reduce_sum(2.*np.pi * getattr(self, 'frequency_' + str(i)) * r , 2)
            k += getattr(self, 'variance_' + str(i)) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.Nc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var



class MaternSpecMixSLS(gpflow.kernels.Kern):
    '''
    Matern spectral mixture kernel with single lengthscale and single variance.
    '''

    def __init__(self, input_dim, lengthscales=None, variances=None, frequencies=None, ideal_f0=None, Nc=1):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        self.ideal_f0 = ideal_f0
        if ideal_f0 == None:
            self.variance = Param(variance, transforms.Logistic(0., 0.25) )
            self.lengthscales = Param(lengthscales, transforms.Logistic(0., 1.0) )
            self.Nc = variances.size
            for i in range(self.Nc): # generate a param object for each  var, and freq, they must be (Nc,) arrays.
                setattr(self, 'variance_' + str(i+1), Param(variances[i], transforms.Logistic(0., 0.25) ) )
                setattr(self, 'frequency_' + str(i+1), Param(frequencies[i], transforms.positive ) )

        else:
            self.variance = Param(variance, transforms.Logistic(0., 0.25) )
            self.lengthscales = Param(0.05, transforms.Logistic(0., 0.1) )
            self.Nc = Nc
            for i in range(Nc): # generate a param object for each len, var, and freq, they must be (Nc,) arrays.
                setattr(self, 'variance_' + str(i+1), Param(0.125, transforms.Logistic(0., 0.25) ) )
                setattr(self, 'frequency_' + str(i+1), Param((i+1)*ideal_f0, transforms.positive ) )

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency_1 * r , 2)
        k = self.variance_1 * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.Nc + 1):
            r1 = tf.reduce_sum(r / self.lengthscales, 2)
            r2 = tf.reduce_sum(2.*np.pi * getattr(self, 'frequency_' + str(i)) * r , 2)
            k += getattr(self, 'variance_' + str(i)) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.Nc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var


class NonParamSpecMix(gpflow.kernels.Kern):
    '''
    Non-parametric spectral mixture kernel
    '''
    pass
    # def __init__(self, input_dim, x, y, fs, variance=1., active_dims=None):
    #     gpflow.kernels.Kern.__init__(self, input_dim, active_dims=active_dims)
    #     self.variance = Param(variance, transforms.positive)
    #     self.ARD = False
    #     self.x, self.y, self.fs = x, y, fs
    #     self.N = y.size
    #     self.Y = fft(y.reshape(-1,)) #  FFT data
    #     self.Yhat = np.abs(self.Y) #  FFT data
    #     self.cov = ifft(self.Yhat)[0:self.N//2]
    #     self.F = np.linspace(0, fs/2., self.N//2) #  frequency vector
    #
    #
    # def K(self, X, X2=None, presliced=False):
    #     if not presliced:
    #         X, X2 = self._slice(X, X2)
    #     if X2 is None:
    #         X2 = X
    #     # Introduce dummy dimension so we can use broadcasting
    #     f = tf.expand_dims(X, 1)  # now N x 1 x D
    #     f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
    #     r = tf.sqrt(tf.square(f - f2))
    #     r = tf.reduce_sum(r, 2)
    #     #index = find   equal values in x array and return index
    #     return self.variance * (r - self.x)
    #
    #
    # def Kdiag(self, X, presliced=False):
    #     return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))
















#
