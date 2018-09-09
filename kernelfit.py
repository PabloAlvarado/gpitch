import gpitch
import numpy as np
import scipy.optimize as opti


def loss_func(p, x, y):
    '''
    Loss function to fit function to kernel observations
    '''
    f = np.sqrt(np.square(approximate_kernel(p, x) - y).mean())
    return f


def approximate_kernel(p, x):
    '''
    approximate kernel
    '''
    nparams = p.size
    npartials = (nparams - 2) / 2
    bias = np.sqrt(p[0] * p[0])

    k_e = (1. + np.sqrt(3.) * np.abs(x) / np.sqrt(p[1] * p[1])) * np.exp(- np.sqrt(3.) * np.abs(x) / np.sqrt(p[1] * p[1]))
    #k_e =  np.exp(- np.sqrt(3.) * np.abs(x) / np.sqrt(p[1] * p[1]))

    k_partials = [np.sqrt(p[i] * p[i]) * np.cos(2 * np.pi * np.sqrt(p[i + npartials] * p[i + npartials]) * np.abs(x))
                  for i in range(2, 2 + npartials)]
    k_fun = 0.*bias + k_e * sum(k_partials)
    return k_fun


def optimize_kern(x, y, p0):
    """Optimization of kernel"""
    phat = opti.minimize(loss_func, p0, method='L-BFGS-B', args=(x, y), tol=1e-16, options={'disp': True})
    pstar = np.sqrt(phat.x ** 2).copy()
    return pstar


def fit(kern, audio, file_name, max_par, fs=44100):
    """Fit kernel to data """

    # time vector for kernel
    n = kern.size
    xkern = np.linspace(0., (n - 1.) / fs, n).reshape(-1, 1)

    # initialize parameters
    if0 = gpitch.find_ideal_f0([file_name])[0]
    init_f, init_v = gpitch.init_cparam(y=audio, fs=fs, maxh=max_par, ideal_f0=if0, scaled=False)[0:2]
    init_l = np.array([0., 1.])

    # optimization
    p0 = np.hstack((init_l, init_v, init_f))  # initialize params
    pstar = optimize_kern(x=xkern, y=kern, p0=p0)

    # compute initial and learned kernel
    kern_init = approximate_kernel(p0, xkern)
    kern_approx = approximate_kernel(pstar, xkern)

    # get kernel hyperparameters
    npartials = (pstar.size - 2) / 2
    noise_var = pstar[0]
    lengthscale = pstar[1]
    variance = pstar[2: npartials + 2]
    frequency = pstar[npartials + 2:]
    params = [lengthscale, variance, frequency]
    return params, kern_init, kern_approx
