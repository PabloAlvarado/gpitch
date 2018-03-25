import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
import modgp
from kernels import Matern12Cosine
from scipy.fftpack import fft
from scipy import signal
import os
import fnmatch
import tensorflow as tf
import peakutils
import soundfile
import pickle
import loogp
import time

def init_vv(x, y, niv, kern_com, kern_act, maxiter=100):
    """
    Initializer of variational variables for the LOO model
    :param x: time vector
    :param y: audio data
    :param niv: number of inducing variables per window
    :param kernels: list with 4 elements [kern_com1, kern_act1, kern_com2, kern_act2]
    :param maxiter: maximun number of iterations per window
    :return: list with q_mu and q_sqrt for each process
    """

    def get_predict(x, y, xnew, m):
        m.X = x
        m.Y = y
        return m.predict_f(xnew)

    l_q_mu1 = []  # list to save variational parameteres
    l_q_mu2 = []
    l_q_mu3 = []
    l_q_mu4 = []
    l_q_sqrt1 = []
    l_q_sqrt2 = []
    l_q_sqrt3 = []
    l_q_sqrt4 = []
    l_z = []

    n = y.size  # size of data
    nsw = 16000  # number samples per window
    nw = n / nsw  # number of windows
    x_win = [x[i * nsw: (i + 1) * nsw].copy() for i in range(nw)]
    y_win = [y[i*nsw : (i+1)*nsw].copy() for i in range(nw)]
    zinit = np.linspace(x_win[0][0], x_win[0][-1], niv).reshape(-1, 1)
    model = loogp.LooGP(X=x_win[0].copy(), Y=y_win[0].copy(), kf=kern_com, kg=kern_act, Z=zinit)

    for i in range(nw):
        model.X = x_win[i].reshape(-1, 1)
        model.Y = y_win[i].reshape(-1, 1)
        model.Z = np.linspace(x_win[i][0], x_win[i][-1], niv).reshape(-1, 1)

        model.q_mu1 = np.zeros((model.Z.value.shape[0], 1))  # f1
        model.q_mu3 = np.zeros((model.Z.value.shape[0], 1))  # f2
        model.q_mu2 = -2.1972 * np.ones((model.Z.value.shape[0], 1))  # g1
        model.q_mu4 = -2.1972 * np.ones((model.Z.value.shape[0], 1))  # g2

        q_sqrt = np.array([np.eye(model.Z.value.shape[0]) for _ in range(1)]).swapaxes(0, 2)

        model.q_sqrt1, model.q_sqrt2, model.q_sqrt3, model.q_sqrt4 = [q_sqrt.copy() for _ in range(4)]

        st = time.time()
        model.optimize(disp=1, maxiter=maxiter)
        print(time.time() - st)

        l_q_mu1.append(model.q_mu1.value)  # f1
        l_q_mu2.append(model.q_mu2.value)  # g1
        l_q_mu3.append(model.q_mu3.value)  # f2
        l_q_mu4.append(model.q_mu4.value)  # g2

        l_q_sqrt1.append(model.q_sqrt1.value)
        l_q_sqrt2.append(model.q_sqrt2.value)
        l_q_sqrt3.append(model.q_sqrt3.value)
        l_q_sqrt4.append(model.q_sqrt4.value)

        l_z.append(model.Z.value)

    m_iv_com1 = gpflow.gpr.GPR(l_z[0].copy(), l_q_mu1[0].copy(), kern_com[0])  # f1
    m_iv_act1 = gpflow.gpr.GPR(l_z[0].copy(), l_q_mu2[0].copy(), kern_act[0])  # g1
    m_iv_com2 = gpflow.gpr.GPR(l_z[0].copy(), l_q_mu3[0].copy(), kern_com[1])  # f2
    m_iv_act2 = gpflow.gpr.GPR(l_z[0].copy(), l_q_mu4[0].copy(), kern_act[1])  # g2


    l_q_mu1_hr = []  # list to save variational parameteres hihg resolution
    l_q_mu2_hr = []
    l_q_mu3_hr = []
    l_q_mu4_hr = []
    l_q_sqrt1_hr = []
    l_q_sqrt2_hr = []
    l_q_sqrt3_hr = []
    l_q_sqrt4_hr = []
    l_z_hr = []
    for i in range(nw):
        x_pred = np.linspace(i, i + 1, 100).reshape(-1, 1)
        mean_com1, var_com1 = get_predict(x=l_z[i], y=l_q_mu1[i], xnew=x_pred, m=m_iv_com1)
        mean_act1, var_act1 = get_predict(x=l_z[i], y=l_q_mu2[i], xnew=x_pred, m=m_iv_act1)
        mean_com2, var_com2 = get_predict(x=l_z[i], y=l_q_mu3[i], xnew=x_pred, m=m_iv_com2)
        mean_act2, var_act2 = get_predict(x=l_z[i], y=l_q_mu4[i], xnew=x_pred, m=m_iv_act2)

        l_q_mu1_hr.append(mean_com1)
        l_q_mu2_hr.append(mean_act1)
        l_q_mu3_hr.append(mean_com2)
        l_q_mu4_hr.append(mean_act2)
        l_q_sqrt1_hr.append(var_com1)
        l_q_sqrt2_hr.append(var_act1)
        l_q_sqrt3_hr.append(var_com2)
        l_q_sqrt4_hr.append(var_act2)
        l_z_hr.append(x_pred)

    q_mu1 = np.asarray(l_q_mu1).reshape(-1, 1)
    q_mu2 = np.asarray(l_q_mu2).reshape(-1, 1)
    q_mu3 = np.asarray(l_q_mu3).reshape(-1, 1)
    q_mu4 = np.asarray(l_q_mu4).reshape(-1, 1)
    #q_sqrt1 = np.asarray(l_q_sqrt1).reshape(-1, 1)
    #q_sqrt2 = np.asarray(l_q_sqrt2).reshape(-1, 1)
    #q_sqrt3 = np.asarray(l_q_sqrt3).reshape(-1, 1)
    #q_sqrt4 = np.asarray(l_q_sqrt4).reshape(-1, 1)


    q_mu1_hr = np.asarray(l_q_mu1_hr).reshape(-1, 1)
    q_mu2_hr = np.asarray(l_q_mu2_hr).reshape(-1, 1)
    q_mu3_hr = np.asarray(l_q_mu3_hr).reshape(-1, 1)
    q_mu4_hr = np.asarray(l_q_mu4_hr).reshape(-1, 1)
    q_sqrt1_hr = np.asarray(l_q_sqrt1_hr).reshape(-1, 1)
    q_sqrt2_hr = np.asarray(l_q_sqrt2_hr).reshape(-1, 1)
    q_sqrt3_hr = np.asarray(l_q_sqrt3_hr).reshape(-1, 1)
    q_sqrt4_hr = np.asarray(l_q_sqrt4_hr).reshape(-1, 1)

    z = np.asarray(l_z).reshape(-1, 1)

    q_mu = [q_mu1, q_mu2, q_mu3, q_mu4]
    q_sqrt = [l_q_sqrt1, l_q_sqrt2, l_q_sqrt3, l_q_sqrt4]

    q_mean_hr = [q_mu1_hr, q_mu2_hr, q_mu3_hr, q_mu4_hr]
    q_var_hr = [q_sqrt1_hr, q_sqrt2_hr, q_sqrt3_hr, q_sqrt4_hr]
    return q_mu, q_sqrt, z, q_mean_hr, q_var_hr




def loadm(directory, pattern=''):
    '''load an already gpitch trained model'''

    #filenames = os.listdir(directory)  # filenames of models to load
    filenames = []
    filenames += [i for i in os.listdir(directory) if pattern in i]
    m_list = []  # list of models loaded
    for i in range(len(filenames)):
        m_list.append(pickle.load(open(directory + filenames[i], "rb")))
    return m_list, filenames


def find_ideal_f0(string):
    """"""
    ideal_f0 = []
    for j in range(len(string)):
        for i in range(21, 89):
            if string[j].find('M' + str(i)) is not -1:
                ideal_f0.append(midi2freq(i))
    return ideal_f0

def readaudio(fname, frames=-1, start=0, aug=False):
    y, fs = soundfile.read(fname, frames=frames, start=start)  # load data and sample freq
    y = y.reshape(-1, 1)
    y /= np.max(np.abs(y))
    y += -y.mean()  # move data to have zero mean and bounded between (-1, 1)
    if aug:
        augnum = 1000  # number of zeros to add
        y = np.append(np.zeros((augnum, 1)), y[0: -augnum]).reshape(-1, 1)
    frames = y.size
    x = np.linspace(0, (frames-1.)/fs, frames).reshape(-1, 1)  # time vector
    return x, y, fs


def init_cparam(y, fs, maxh, ideal_f0, scaled=True, win_size=10):
    '''
    :param y: data
    :param fs: sample frequency
    :param maxh: max number of partials
    :param ideal_f0: ideal f0 or pitch
    :param scaled: to scale or not the variance
    :param win_size: size of window to smooth spectrum
    :return:
    '''

    N = y.size
    Y = fft(y.reshape(-1,)) #  FFT data
    S =  2./N * np.abs(Y[0:N//2]) #  spectral density data
    F = np.linspace(0, fs/2., N//2) #  frequency vector

    win =  signal.hann(win_size)
    Ss = signal.convolve(S, win, mode='same') / sum(win)
    Sslog = np.log(Ss)
    Sslog = Sslog + np.abs(np.min(Sslog))
    Sslog /= np.max(Sslog)
    thres = 0.10*np.max(Sslog)
    min_dist = 0.8*np.argmin(np.abs(F - ideal_f0))
    idx = peakutils.indexes(Sslog, thres=thres, min_dist=min_dist)

    F_star, S_star = F[idx], Ss[idx]

    idx_sorted = np.argsort(F_star.copy())
    S_star = S_star[idx_sorted]
    F_star = np.sort(F_star)


    for index in range(F_star.size):
        if F_star[index] < 0.75*ideal_f0:
            F_star2 = np.delete(F_star, [index])
            S_star2 = np.delete(S_star, [index])

    aux1 = np.flip(np.sort(S_star2), 0)
    aux2 = np.flip(np.argsort(S_star2), 0)

    if aux1.size > maxh :
        vvec = aux1[0:maxh]
        idxf = aux2[0:maxh]
    else :
        vvec = aux1
        idxf = aux2

    if scaled:
        sig_scale = 1./ (4.*np.sum(vvec)) #rescale (sigma)
        vvec *= sig_scale

    freq_final = F_star2[idxf]
    var_final = vvec

    idx_sorted = np.argsort(freq_final.copy())
    var_final = var_final[idx_sorted]
    freq_final = np.sort(freq_final)

    return [freq_final, var_final, F, Ss, thres]


def init_settings(visible_device, interactive=False):
    '''Initialize usage of GPU and plotting
       visible_device : which GPU to use'''

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # deactivate tf warnings (default 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device  # configuration use only one GPU
    config = tf.ConfigProto()  # configuration to not to use all the memory
    config.gpu_options.allow_growth = True
    if interactive == True:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def load_filenames(directory, pattern, bounds):
    auxl = fnmatch.filter(os.listdir(directory), pattern)
    filel = [fnmatch.filter(auxl, '*_M' + str(pitch) + '_*')[0]
             for pitch in range(bounds[0], bounds[1]) ]
    filel =  np.asarray(filel).reshape(-1,)
    return filel


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


# def init_com_params(y, fs, Nh, ideal_f0, scaled=True, win_size=6):
#     N = y.size
#     Y = fft(y.reshape(-1,)) #  FFT data
#     S =  2./N * np.abs(Y[0:N//2]) #  spectral density data
#     F = np.linspace(0, fs/2., N//2) #  frequency vector
#     win =  signal.hann(win_size)
#     Ss = signal.convolve(S, win, mode='same') / sum(win)
#     #Ss /= np.max(Ss)
#
#     F_star = np.zeros((Nh,))
#     S_star = np.zeros((Nh,))
#
#     f0 = ideal_f0
#     for i in range(Nh):
#         S_hat = Ss.copy()
#         S_hat[F <= (i+0.5)*f0] = 0.
#         S_hat[F >= (i+1.5)*f0] = 0.
#         idx_max = np.argmax(S_hat)
#         F_star[i] = F[idx_max]
#         S_star[i] = Ss[idx_max]
#         if i == 0:
#             f0 = F_star[i].copy()  # update value of natural frequency
#
#     if scaled:
#         sig_scale = 1./ (4.*np.sum(S_star)) #rescale (sigma)
#         S_star *= sig_scale
#     return F_star, S_star, F, Y, Ss

def norm(x):
    """divide by absolute max"""
    return x / np.max(np.abs(x))


def logistic(x):
    """ logistic function """
    return 1./(1. + np.exp(-x))

def ilogistic(x):
    """inverse logistic function"""
    return - np.log(1./x - 1.)

def softplus(x):
    """ softplus function """
    return np.log(np.exp(x) + 1.)

def isoftplus(x):
    """ inverse softplus function """
    return np.log(np.exp(x) - 1.)

def logistic_tf(x):
    """logistic function using tensorflow """
    return 1./(1. + tf.exp(-x))

def softplus_tf(x):
    """ softplus function using tensorflow  """
    return tf.log(tf.exp(x) + 1.)

def isoftplus_tf(x):
    """ inverse softplus function using tensorflow  """
    return tf.log(tf.exp(x) - 1.)

def ilogistic_tf(x):
    """inverse logistic function using tensorflow"""
    return - tf.log(1./x - 1.)


def Matern12CosineMix(variance, lengthscale, period, Nh):
    '''Write it.'''
    kern_list = [Matern12Cosine(input_dim=1, period=period[i], variance=variance[i], lengthscales=lengthscale[i]) for i in range(0, Nh)]
    return gpflow.kernels.Add(kern_list)


def wavread(filename, start=0, N=None, norm=True, mono=False):
    """Load .wav audio file."""
    fs, y = wav.read(filename)
    y = y.astype(np.float64)
    if mono:
        y = np.mean(y, 1)
    if norm:
        y = y / np.max(np.abs(y))

    if N == None:
    	N = y.size
    y = y[start: N + start].reshape(-1, 1) # select data subset
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
    return x, y, fs


def load_pitch_params_data(pitch_list, data_loc, params_loc):
    '''
    This function loads the desired pitches and the gets the names of the files in the MAPS dataset
    corresponding to those pitches. Also returns the learned params and data related to
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


def midi2freq(midi):
    return 2.**( (midi - 69.)/12. ) * 440.

def freq2midi(freq):
    return int(69. + 12. * np.log2(freq / 440.))
































#
