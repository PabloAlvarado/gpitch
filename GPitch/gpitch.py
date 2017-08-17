import numpy as np
import scipy as sp
import GPflow
import modgp


def Lorentzian(p, x):
    '''
    Lorentzian function http://mathworld.wolfram.com/LorentzianFunction.html
    '''
    return p[0]*p[1]/(4.*np.square(np.pi)*(x - p[2]/(2.*np.pi))**2. + p[1]**2.)


def Lloss(p, x, y):
    '''
    Loss function to fit a Lorentzian function to data "y"
    '''
    f =  np.sqrt(np.square(Lorentzian(p, x) - y).mean())
    return f


def LorM(x, s, l, f):
    '''
    Mixture of Lorentzian functions
    '''
    lm = np.zeros((x.shape))
    for i in range(0, s.size):
        lm += s[i]*l[i] / ( (4.*np.square(np.pi))*(x - f[i]/(2.*np.pi)) **2. + \
              l[i]**2. )
    return lm


def MaternSM(x, s, l, f):
    '''
    Matern spectral mixture function
    '''
    ker = np.zeros((x.shape))
    for i in range(0, s.size):
        ker += s[i] * np.exp(-l[i]*np.abs(x)) * np.cos(f[i]*x)
    return ker


def ker_msm(s, l, f, Nh):
    '''
    Matern spectral mixture kernel
    Input:
    s  : variance vector
    l  : Matern lengthscales vector
    f  : frequency vector (Hz)
    Nh : number of components

    Output:
    GPflow kernel object
    '''
    per = 1./(2.*np.pi*f)
    kexp0 = GPflow.kernels.Matern12(input_dim=1, variance=1.0,
                                    lengthscales=l[0])
    kcos0 = GPflow.kernels.Cosine(input_dim=1, variance=s[0],
                                  lengthscales=per[0])
    ker = kexp0*kcos0

    for n in range (1, Nh):
        kexp = GPflow.kernels.Matern12(input_dim=1, variance=1.0,
                                       lengthscales=l[n])
        kcos = GPflow.kernels.Cosine(input_dim=1, variance=s[n],
                                     lengthscales=per[n])
        ker += kexp*kcos
    return ker


def learnparams(X, S, Nh):
    '''
    Learn parameters in frequency domain.
    Input:
    X: frequency vector (Hz)
    S: Magnitude Fourier transform signal
    Nh: number of maximun harmonic to learn

    Output:
    matrix of parameters
    '''
    Np = 3 # number of parameters per Lorentzian
    Pstar = np.zeros((Nh,Np))
    Shat = S.copy()
    #plt.figure()
    count = 0
    for i in range(0, Nh):
        idx = np.argmax(Shat)
        if Shat[idx] > 0.010*S.max():
            count += 1
            a = idx - 25
            if a < 0:
                a = 0
            b = idx + 25
            x = X
            y = Shat
            p0 = np.array([1., 10., 2.*np.pi*X[idx]])
            phat = sp.optimize.minimize(Lloss, p0, method='L-BFGS-B',
                                        args=(x, y), tol=1e-10,
                                        options={'disp': False})
            pstar = phat.x
            Pstar[i,:] = pstar
            learntfun = Lorentzian(pstar, x)
            Shat = np.abs(learntfun -  Shat)
            Shat[a:b,] = 0.
            #plt.subplot(1,Nh,i+1)
            #plt.plot(x[a:b],S[a:b],'.b', ms = 3)
            #plt.plot(x[a:b],learntfun[a:b],'g', lw = 1)
            #plt.axis('off')
            #plt.ylim([S.min(), S.max()])
    s_s, l_s, f_s = np.hsplit(Pstar[0:count,:], 3)
    return s_s, 1./l_s, f_s/(2.*np.pi),


def logistic(x):
    return 1./(1+ np.exp(-x))


def myplot():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.115, bottom=.12, right=.99, top=.97)
    fig.set_size_inches(width, height)
    plt.grid(False)


# Gaussian Process pitch detection
def gppd(x=None, y=None, ws=None, d=None, nc=None, nu=None):
    '''
    Gaussian process (multi)pitch detection.
    INPUTS:
    x  = time column vector ndarray
    y  = audio vector ndarray
    ws = size of windows to perform analysis
    d  = number of pitches (latent sources)
    nc = number of frequency components for kernels quasi-periodic processes
    nu = number of inducing points

    OUTPUT:
    q_params = list with ws elements, where each element is a list with 2D
    elements, that is D mean vectors and the corresponding D varianace vectors.
    '''
    q_params = [np.linspace(1,2,i) for i in range(1, 4)]
    return q_params

























#
