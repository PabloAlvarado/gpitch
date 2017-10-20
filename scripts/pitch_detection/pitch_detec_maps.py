import sys, os
sys.path.append('../../')
import matplotlib
if True: matplotlib.use('agg') # define if running code on server (True)
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import gpflow, gpitch
from gpitch.amtgp import logistic


visible_device = sys.argv[1] #  load external variable (gpu to use)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #  deactivate tf warnings
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True) #  confi gpu usage, plot

data_location = '../../../datasets/maps/sample_rate_16khz/'
params_location = '../../../results/files/params_activations/'
test_data_location = '../../../datasets/maps/test_data/'
results_files_location = '../../../results/files/pitch_detection/'
results_figures_location = '../../../results/figures/pitch_detection/'

intensity = 'F' #  property maps datset, choose "forte" sounds
pitch_list = np.asarray(['60', '64', '67', '72', '76']) #  pitches to detect
Np = pitch_list.size
filename_list =[None]*Np
lfiles = gpitch.amtgp.load_filename_list(data_location + 'filename_list.txt')
j = 0
for pitch in pitch_list:
    for i in lfiles:
        if pitch in i:
            if intensity in i:
                filename_list[j] = i
                j += 1
final_list  = np.asarray(filename_list).reshape(-1, )
print final_list

train_data = [None]*Np #  load training data and learned params
params = [None]*Np
for i in range(Np):
    N = 32000 # numer of data points to load
    fs, aux = gpitch.amtgp.wavread(data_location + final_list[i] + '.wav', start=5000, N=N)
    train_data[i] = aux.copy()
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
    params[i] = np.load(params_location + 'params_act_' + final_list[i] + '.npz')
    keys = np.asarray(params[i].keys()).reshape(-1,)


kern_f_list = [None]*Np #  set kernels
kern_g_list = [None]*Np
activations = [None]*Np
components =  [None]*Np
for i in range(Np):
    print i
    kern_f_list[i] = gpitch.amtgp.Matern12CosineMix(variance=params[i]['s_com'],
                                                    lengthscale=params[i]['l_com'],
                                                    period=1./params[i]['f_com'],
                                                    Nh=params[i]['Nc'])
    kern_g_list[i] = gpflow.kernels.Matern32(input_dim=1, variance=params[i]['s_act'],
                                             lengthscales=params[i]['l_act'])

test_data_name = test_data_location + 'test_data_5_pitches.wav'
fs, y_test = gpitch.amtgp.wavread(test_data_name, mono=False) # load test_data


Xtest = x.copy()
ytest = y_test.copy()
X = x.copy()

Ns = 16000
winit = 0 #initial window to analyse
wfinish = Xtest.size/Ns # final window to analyse
wfinish = 1
Nw = wfinish - winit # number of windows to analyse
# initialize arrays to save results
save_X = np.zeros((Ns,Nw))
save_y = np.zeros((Ns,Nw))
# comp and act 1
save_muf1 = np.zeros((Ns,Nw))
save_varf1 = np.zeros((Ns,Nw))
save_mug1 = np.zeros((Ns,Nw))
save_varg1 = np.zeros((Ns,Nw))
# comp and act 2
save_muf2 = np.zeros((Ns,Nw))
save_varf2 = np.zeros((Ns,Nw))
save_mug2 = np.zeros((Ns,Nw))
save_varg2 = np.zeros((Ns,Nw))

noise_var = 1e-7
count = 0 # count number of windows analysed so far
jump = 160

X = Xtest[0:Ns]
y = ytest[0:Ns]
#Z = X[::jump].copy()
Z = np.vstack(( (X[::jump].copy()).reshape(-1,1) ,(X[-1].copy()).reshape(-1,1)  ))
#m = siggp.ModGP(X, y, k_w3, k_loo, k_g1, k_g2, Z)
k_loo = kern_f_list[1] + kern_f_list[2] + kern_f_list[3] + kern_f_list[4]
m = gpitch.loogp.LooGP(X, y,
                       kern_f_list[0], k_loo, kern_g_list[0], kern_g_list[0],
                       Z)
m.kern1.fixed = True
m.kern2.fixed = True
m.kern3.fixed = True
m.kern4.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True

for i in range (winit, wfinish):
    count = count + 1

    X = Xtest[ i*Ns : (i+1)*Ns]
    y = ytest[ i*Ns : (i+1)*Ns]
    Z = X[::jump].copy()
    #Z = np.vstack(( (X[::jump].copy()).reshape(-1,1) ,(X[-1].copy()).reshape(-1,1)  ))

    m.X = X.copy()
    m.Y = y.copy()
    m.Z = Z.copy()

    m.q_mu1._array = y[::jump].copy()
    m.q_mu2._array = y[::jump].copy()
    m.q_mu3._array = np.zeros(Z.shape)
    m.q_mu4._array = np.zeros(Z.shape)

    m.q_sqrt1._array = np.expand_dims(np.eye(Z.size), 2)
    m.q_sqrt2._array = np.expand_dims(np.eye(Z.size), 2)
    m.q_sqrt3._array = np.expand_dims(np.eye(Z.size), 2)
    m.q_sqrt4._array = np.expand_dims(np.eye(Z.size), 2)

    print('Analysing window number ', count, ' total number of windows to analyse ', Nw)
    m.optimize(disp=1, maxiter = 200)

    mu_f1, var_f1 = m.predict_f1(X)
    mu_f2, var_f2 = m.predict_f2(X)
    mu_g1, var_g1 = m.predict_g1(X)
    mu_g2, var_g2 = m.predict_g2(X)

    save_X[:,i-winit] = X.reshape(-1)
    save_y[:,i-winit] = y.reshape(-1)
    save_muf1[:,i-winit] = mu_f1.reshape(-1)
    save_muf2[:,i-winit] = mu_f2.reshape(-1)
    save_mug1[:,i-winit] = mu_g1.reshape(-1)
    save_mug2[:,i-winit] = mu_g2.reshape(-1)
    save_varf1[:,i-winit] = var_f1.reshape(-1)
    save_varf2[:,i-winit] = var_f2.reshape(-1)
    save_varg1[:,i-winit]= var_g1.reshape(-1)
    save_varg2[:,i-winit]= var_g2.reshape(-1)
    #ll = m.compute_log_likelihood()
    #save_ll[i-winit,0] =  ll

X = np.reshape(save_X, (-1,1), order = 'F')
y = np.reshape(save_y, (-1,1), order = 'F')
mu_f1 = np.reshape(save_muf1, (-1,1), order = 'F')
mu_f2 = np.reshape(save_muf2, (-1,1), order = 'F')
mu_g1 = np.reshape(save_mug1, (-1,1), order = 'F')
mu_g2 = np.reshape(save_mug2, (-1,1), order = 'F')
var_f1 = np.reshape(save_varf1, (-1,1), order = 'F')
var_f2 = np.reshape(save_varf2, (-1,1), order = 'F')
var_g1 = np.reshape(save_varg1, (-1,1), order = 'F')
var_g2 = np.reshape(save_varg2, (-1,1), order = 'F')

np.savez_compressed(results_location + 'pitch_detection_5_pitches',
                              X = x,
                              y = y_test,
                              mu_f1 = mu_f1, var_f1 = var_f1,
                              mu_f2 = mu_f2, var_f2 = var_f2,
                              mu_g1 = mu_g1, var_g1 = var_g1,
                              mu_g2 = mu_g2, var_g2 = var_g2)

print 'Go!'
