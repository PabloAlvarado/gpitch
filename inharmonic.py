import tensorflow as tf
from gpflow import kernels, param, transforms


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
