import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
import modgp
from kernels import Matern12Cosine
from scipy.fftpack import fft
from scipy import signal
import os
import tensorflow as tf

def init_settings(visible_device='0', interactive=False):
    '''Initialize usage of GPU and plotting'''
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #  deactivate tf warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device # configuration use only one GPU
    config = tf.ConfigProto() #Configuration to not to use all the memory
    config.gpu_options.allow_growth = True
    if interactive == True:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)



def load_filename_list(filename):
    file = open(filename, "r")
    list_files =  file.readlines()
    list_files = np.asarray(list_files).reshape(-1,)
    for i in range(list_files.size):
        list_files[i] = list_files[i].strip('.wav\n')
    file.close()
    return list_files


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
    s_s, l_s, f_s = np.hsplit(Pstar[0:count,:], 3)
    return s_s, 1./l_s, f_s/(2.*np.pi),


def init_com_params(y, fs, Nh, ideal_f0, scaled=True):
    N = y.size
    Y = fft(y.reshape(-1,)) #  FFT data
    S =  2./N * np.abs(Y[0:N//2]) #  spectral density data
    F = np.linspace(0, fs/2., N//2) #  frequency vector
    win =  signal.hann(6)
    Ss = signal.convolve(S, win, mode='same') / sum(win)
    #Ss /= np.max(Ss)

    F_star = np.zeros((Nh,))
    S_star = np.zeros((Nh,))
    for i in range(Nh):
        S_hat = Ss.copy()
        S_hat[F <= (i+0.5)*ideal_f0] = 0.
        S_hat[F >= (i+1.5)*ideal_f0] = 0.
        idx_max = np.argmax(S_hat)
        F_star[i] = F[idx_max]
        S_star[i] = Ss[idx_max]

    if scaled:
        sig_scale = 1./ (4.*np.sum(S_star)) #rescale (sigma)
        S_star *= sig_scale
    return F_star, S_star, F, Y, Ss


def logistic(x):
    return 1./(1+ np.exp(-x))


def Matern12CosineMix(variance, lengthscale, period, Nh):
    '''Write it.'''
    kern_list = [Matern12Cosine(input_dim=1, period=period[i], variance=variance[i], lengthscales=lengthscale[i]) for i in range(0, Nh)]
    return gpflow.kernels.Add(kern_list)


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


def load_pitch_params_data(pitch_list, data_loc, params_loc):
    '''
    This function loads the desired pitches and the gets the names of the files in the MAPS dataset
    corresponding to those pitchescthat pitch. Also returns the learned params and data related to
    those files.
    '''
    intensity = 'F'  # property maps datset, choose "forte" sounds
    Np = pitch_list.size # number of pitches
    filename_list =[None]*Np
    lfiles = load_filename_list(data_loc + 'filename_list.txt')
    j = 0
    for pitch in pitch_list:
        for i in lfiles:
            if pitch in i:
                if intensity in i:
                    filename_list[j] = i
                    j += 1
    final_list  = np.asarray(filename_list).reshape(-1, )
    train_data = [None]*Np #  load training data and learned params
    params = [None]*Np
    for i in range(Np):
        N = 32000 # numer of data points to load
        fs, aux = wavread(data_loc + final_list[i] + '.wav', start=5000, N=N)
        train_data[i] = aux.copy()
        x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
        params[i] = np.load(params_loc + 'params_act_' + final_list[i] + '.npz')
        keys = np.asarray(params[i].keys()).reshape(-1,)
    return final_list, train_data, params



































#
