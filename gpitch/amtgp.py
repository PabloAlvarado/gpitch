import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
import modgp
from kernels import Matern12Cosine
from scipy.fftpack import fft
import os
import tensorflow as tf
from matplotlib import pyplot as plt

def init_settings(visible_device='0', interactive=False):
    '''Initialize usage of GPU and plotting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device # configuration use only one GPU
    config = tf.ConfigProto() #Configuration to not to use all the memory
    config.gpu_options.allow_growth = True
    if interactive == True:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)

    # plotting initial configuration
    plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
    plt.interactive(True)
    plt.close('all')
    

def Lorentzian(p, x):
    '''
    Lorentzian function http://mathworld.wolframodel.com/LorentzianFunction.html
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
    Matern spectral mixture kernel in gpflow
    Input:
    s  : variance vector
    l  : Matern lengthscales vector
    f  : frequency vector (Hz)
    Nh : number of components

    Output:
    gpflow kernel object
    '''
    per = 1./(2.*np.pi*f)
    kexp0 = gpflow.kernels.Matern12(input_dim=1, variance=1.0,
                                    lengthscales=l[0])
    kcos0 = gpflow.kernels.Cosine(input_dim=1, variance=s[0],
                                  lengthscales=per[0])
    ker = kexp0*kcos0

    for n in range (1, Nh):
        kexp = gpflow.kernels.Matern12(input_dim=1, variance=1.0,
                                       lengthscales=l[n])
        kcos = gpflow.kernels.Cosine(input_dim=1, variance=s[n],
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
        if Shat[idx] > 0.02*S.max():
            count += 1
            a = idx - 25
            b = idx + 25
            if a < 0:
                a = 0
            x = X
            y = Shat
            p0 = np.array([1.0, 0.1, 2.*np.pi*X[idx]])
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



def Matern12CosineMix(variance, lengthscale, period, Nh):
    '''Write it.'''
    kern_list = [Matern12Cosine(input_dim=1, period=period[i], variance=variance[i], lengthscales=lengthscale[i]) for i in range(0, Nh)]
    return gpflow.kernels.Add(kern_list)



def myplot():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.115, bottom=.12, right=.99, top=.97)
    fig.set_size_inches(width, height)
    plt.grid(False)


def wavread(filename, start=0, N=None, norm=True, mono=True):
    fs, y = wav.read(filename)
    y = y.astype(np.float64)
    if mono:
        y = np.mean(y, 1)
    if norm:
        y = y / np.max(np.abs(y))

    if N == None:
    	N = y.size
    y = y[start: N + start].reshape(-1, 1) # select data subset
    return fs, y



# Gaussian Process pitch detection using modulated GP
# class ModPDet():
#     def __init__(self, x, y, fs, ws, jump):
#         self.x, self.y = x, y
#         self.fs, self.ws = fs, ws # sample rate, and window size (samples)
#         self.jump, self.N = jump, x.size
#         self.Nw = self.N/self.ws  # number of windows
#         self.Nh = 10 # number of maximun frequency components in density
#
#         # frequency representation data
#         self.Y = fft(y.reshape(-1,)) #  FFT data
#         self.S =  2./self.N * np.abs(self.Y[0:self.N/2]) #  spectral density data
#         self.F = np.linspace(0, fs/2., self.N/2) #  frequency vector
#
#         self.s, self.l, self.f = learnparams(X=self.F, S=self.S, Nh=self.Nh) #  Param learning Nh=#harmonics
#         sig_scale = 1./ (4.*np.sum(self.s)) #rescale (sigma)
#         self.s *= sig_scale
#
#         self.kf = Matern12CosineMix(variance=self.s, lengthscale=self.l, period=1./self.f, Nh=self.s.size)
#         self.kg = gpflow.kernels.Matern32(input_dim=1, variance=10.*np.random.rand(), lengthscales=10*np.random.rand())
#
#         self.x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)] # split data into windows
#         self.y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, self.Nw)]
#         self.z = self.x_l[0][::jump].copy()
#
#         self.model = modgp.ModGP(self.x_l[0].copy(), self.y_l[0].copy(), self.kf, self.kg, self.z, whiten=True)
#         self.model.likelihood.noise_var = 1e-7
#         self.model.likelihood.noise_var.fixed = True
#         self.model.q_mu1.transform = gpflow.transforms.Logistic(a=-1.0, b=1.0)
#         self.model.kern1.fixed = True # component kernel
#         self.model.kern2.fixed = True # activation kernel
#
#     def optimize(self, disp, maxiter):
#
#         init_list = [np.zeros(self.model.Z.shape) for i in range(0, self.Nw)]  # list to save predictions mean (qm) and variance (qv)
#         self.qm1 = list(init_list)
#         self.qm2 = list(init_list)
#         self.qv1 = list(init_list)
#         self.qv2 = list(init_list)
#         self.x_pred = [np.zeros(self.x_l[0].shape) for i in range(0, self.Nw)]
#         self.y_pred = [np.zeros(self.x_l[0].shape) for i in range(0, self.Nw)]
#
#         for i in range(self.Nw):
#             self.model.X = self.x_l[i].copy()
#             self.model.Y = self.y_l[i].copy()
#             self.model.Z = self.x_l[i][::self.jump].copy()
#             self.model.q_mu1._array = np.zeros(self.model.Z.shape)
#             self.model.q_mu2._array = np.zeros(self.model.Z.shape)
#             self.model.q_sqrt1._array = np.expand_dims(np.eye(self.model.Z.size), 2)
#             self.model.q_sqrt2._array = np.expand_dims(np.eye(self.model.Z.size), 2)
#             self.model.optimize(disp=disp, maxiter=maxiter)
#             self.qm1[i], self.qv1[i] = self.model.predict_f(self.x_l[i])
#             self.qm2[i], self.qv2[i] = self.model.predict_g(self.x_l[i])
#             self.x_pred[i] = self.x_l[i].copy()
#             self.y_pred[i] = self.y_l[i].copy()
#
#         self.qm1 = np.asarray(self.qm1).reshape(-1, 1)
#         self.qm2 = np.asarray(self.qm2).reshape(-1, 1)
#         self.qv1 = np.asarray(self.qv1).reshape(-1, 1)
#         self.qv2 = np.asarray(self.qv2).reshape(-1, 1)
#         self.x_pred = np.asarray(self.x_pred).reshape(-1, 1)
#         self.y_pred = np.asarray(self.y_pred).reshape(-1, 1)
#
#
#     def plot_results(self, zoom_limits):
#         x = self.x_pred
#         y = self.y_pred
#         fig, fig_array = plt.subplots(3, 2, sharex=False, sharey=False)
#
#         fig_array[0, 0].set_title('Data')
#         fig_array[0, 0].plot(x, y, lw=2)
#         fig_array[0, 1].set_title('Approximation')
#         fig_array[0, 1].plot(x, logistic(self.qm2)*self.qm1 , lw=2)
#
#         fig_array[1, 0].set_title('Component')
#         fig_array[1, 0].plot(x, self.qm1, color='C0', lw=2)
#         fig_array[1, 0].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
#                              self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)
#
#         fig_array[1, 1].set_title('Component (zoom in)')
#         fig_array[1, 1].plot(x, self.qm1, color='C0', lw=2)
#         fig_array[1, 1].fill_between(x[:, 0], self.qm1[:, 0] - 2*np.sqrt(self.qv1[:, 0]),
#                              self.qm1[:, 0] + 2*np.sqrt(self.qv1[:, 0]), color='C0', alpha=0.2)
#         fig_array[1, 1].set_xlim(zoom_limits)
#
#         fig_array[2, 0].set_title('Activation')
#         fig_array[2, 0].plot(x, logistic(self.qm2), color='g', lw=2)
#         fig_array[2, 0].fill_between(x[:, 0], logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
#                              logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)
#
#         fig_array[2, 1].set_title('Activation (zoom in)')
#         fig_array[2, 1].plot(x, logistic(self.qm2), 'g', lw=2)
#         fig_array[2, 1].fill_between(x[:, 0], logistic(self.qm2[:, 0] - 2*np.sqrt(self.qv2[:, 0])),
#                              logistic(self.qm2[:, 0] + 2*np.sqrt(self.qv2[:, 0])), color='g', alpha=0.2)
#         fig_array[2, 1].set_xlim(zoom_limits)
#
#         return fig, fig_array















#
