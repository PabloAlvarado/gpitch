# integral solution using standard scipy integrate functions
import numpy as np
from scipy import integrate
import itertools


def logistic(x):
    return 1. / (1. + np.exp(-x))

def ground_truth(z, u_x, u_y, v_x, v_y, v_noise):
    def p_x(x):
        return 1./np.sqrt(2. * np.pi * v_x) * np.exp(- 0.5/v_x*(x - u_x)**2)

    def p_y(y):
        return 1./np.sqrt(2. * np.pi * v_y) * np.exp(- 0.5/v_y*(y - u_y)**2)

    def log_lik(x, y):
        return -0.5*np.log(2.*np.pi) - \
                0.5*np.log(v_noise)  - \
                0.5/v_noise*(z - logistic(x)*y)**2

    def f(x,y):
        return  log_lik(x, y) * p_x(x) * p_y(y)

    result, error =  integrate.nquad(f, [[-np.inf, np.inf], [-np.inf, np.inf]])
    return result, error

def hermgauss(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w

def mvhermgauss(H, D):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.
    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])
    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


























#
