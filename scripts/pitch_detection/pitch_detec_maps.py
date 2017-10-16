import numpy as np
import tensorflow as tf
import gpflow
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import matplotlib
server = False # define if running code on server
if server:
   matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys
import os
sys.path.append('../../')
import gpitch


visible_device = sys.argv[1] #  load external variable (gpu to use)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #  deactivate tf warnings
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=True) #  configure gpu usage and plotting


data_location = '../../../datasets/maps/sample_rate_16khz/'
test_data_location = '../../../datasets/maps/test_data/'
results_location = '../../../results/files/pitch_detection/'

intensity = 'F' #  property maps datset, choose "forte" sounds
pitch_list = np.asarray(['60', '64', '67', '72', '76']) #  pitches to detect
Np = pitch_list.size
filename_list =[None]*Np

location = "../../../datasets/maps/sample_rate_16khz/" # load list of files no analyse
lfiles = gpitch.amtgp.load_filename_list(location + 'filename_list.txt')

j = 0
for pitch in pitch_list:
    for i in lfiles:
        if pitch in i:
            if intensity in i:
                filename_list[j] = i
                j += 1
final_list  = np.asarray(filename_list).reshape(-1, )
print final_list

train_data = [None]*Np

for i in range(Np):
    sf, aux = wav.read(location + final_list[i] + '.wav') #load test data
    aux = aux.astype(np.float64)
    aux = np.mean(aux, 1)
    aux = aux / np.max(np.abs(aux))
    train_data[i] = aux.copy()
    plt.figure()
    plt.plot(aux)

#
# N = np.size(y)
# y = y.astype(np.float64)
# y = y / np.max(np.abs(y))
# X = np.linspace(0, (N-1.)/sf, N)
# yt1 = yt1[0: 2*sf].copy() # set training data pitch 1
# yt2 = yt2[0: 2*sf].copy() # set training data pitch 2
# yt3 = yt3[0: 2*sf].copy() # set training data pitch 3
# yt4 = yt4[0: 2*sf].copy() # set training data pitch 4
# yt5 = yt5[0: 2*sf].copy() # set training data pitch 4
#
# Nt = yt1.size # Number of sample points for training
# Xt = np.linspace(0, (Nt-1.)/sf, Nt)
# Xt1 = X[0:2*sf].copy()
# Xt2 = X[0:2*sf].copy()
# Xt3 = X[0:2*sf].copy()
# Xt4 = X[0:2*sf].copy()
# Xt5 = X[0:2*sf].copy()
#
# Xtest = X.reshape(-1,1)
# ytest = y.reshape(-1,1)
#
#
# y1F, y2F, y3F, y4F, y5F = sp.fftpack.fft(yt1), sp.fftpack.fft(yt2), sp.fftpack.fft(yt3),
#sp.fftpack.fft(yt4), sp.fftpack.fft(yt5) #FT training data
# T = 1.0 / sf # sample spacing
# F = np.linspace(0.0, 1.0/(2.0*T), Nt/2)
# S1 = 2.0/Nt * np.abs(y1F[0:Nt/2]) # spectral density training data 1
# S2 = 2.0/Nt * np.abs(y2F[0:Nt/2]) # spectral density training data 2
# S3 = 2.0/Nt * np.abs(y3F[0:Nt/2]) # spectral density training data 3
# S4 = 2.0/Nt * np.abs(y4F[0:Nt/2]) # spectral density training data 3
# S5 = 2.0/Nt * np.abs(y5F[0:Nt/2]) # spectral density training data 3
#
# # Parameters learning
# Nh = 10 # max num of harmonics allowed
# s1, l1, f1 = gpm.learnparams(X=F, S=S1, Nh=Nh)
# s2, l2, f2 = gpm.learnparams(X=F, S=S2, Nh=Nh)
# s3, l3, f3 = gpm.learnparams(X=F, S=S3, Nh=Nh)
# s4, l4, f4 = gpm.learnparams(X=F, S=S4, Nh=Nh)
# s5, l5, f5 = gpm.learnparams(X=F, S=S5, Nh=Nh)
# par1 = [s1, l1, f1]
# par2 = [s2, l2, f2]
# par3 = [s3, l3, f3]
# par4 = [s4, l4, f4]
# par5 = [s5, l5, f5]
#
# p1, p2, p3, p4, p5 = 1./f1, 1./f2, 1./f3, 1./f4, 1./f5
#
# # define kernel component and activation pitch 1
# # k_w1 = gpm.ker_msm(s=s1, l=l1, f=f1, Nh=s1.size)
# # k_w2 = gpm.ker_msm(s=s2, l=l2, f=f2, Nh=s2.size)
# # k_w3 = gpm.ker_msm(s=s3, l=l3, f=f3, Nh=s3.size)
# # k_w4 = gpm.ker_msm(s=s4, l=l4, f=f4, Nh=s4.size)
# # k_w5 = gpm.ker_msm(s=s5, l=l5, f=f5, Nh=s5.size)
# k_w1 = gpm.Matern12CosineMix(variance=s1, lengthscale=l1, period=p1, Nh=s1.size)
# k_w2 = gpm.Matern12CosineMix(variance=s2, lengthscale=l2, period=p2, Nh=s2.size)
# k_w3 = gpm.Matern12CosineMix(variance=s3, lengthscale=l3, period=p3, Nh=s3.size)
# k_w4 = gpm.Matern12CosineMix(variance=s4, lengthscale=l4, period=p4, Nh=s4.size)
# k_w5 = gpm.Matern12CosineMix(variance=s5, lengthscale=l5, period=p5, Nh=s5.size)
# k_g1 = GPflow.kernels.Matern32(input_dim=1, variance=3.0, lengthscales=0.09)
# k_g2 = GPflow.kernels.Matern32(input_dim=1, variance=3.0, lengthscales=0.09)
#
#
#
# ### Pitch detection
# Ns = 5*1600 #number of samples per window
# #Ns = 8192
# Ns = 16000
# winit = 0 #initial window to analyse
# wfinish = Xtest.size/Ns # final window to analyse
# wfinish = 1
# Nw = wfinish - winit # number of windows to analyse
# # initialize arrays to save results
# save_X = np.zeros((Ns,Nw))
# save_y = np.zeros((Ns,Nw))
# # comp and act 1
# save_muf1 = np.zeros((Ns,Nw))
# save_varf1 = np.zeros((Ns,Nw))
# save_mug1 = np.zeros((Ns,Nw))
# save_varg1 = np.zeros((Ns,Nw))
# # comp and act 2
# save_muf2 = np.zeros((Ns,Nw))
# save_varf2 = np.zeros((Ns,Nw))
# save_mug2 = np.zeros((Ns,Nw))
# save_varg2 = np.zeros((Ns,Nw))
#
# noise_var = 1e-5
# count = 0 # count number of windows analysed so far
# jump = 80
# X = Xtest[0:Ns]
# y = ytest[0:Ns]
# #Z = X[::jump].copy()
# Z = np.vstack(( (X[::jump].copy()).reshape(-1,1) ,(X[-1].copy()).reshape(-1,1)  ))
# #m = siggp.ModGP(X, y, k_w3, k_loo, k_g1, k_g2, Z)
# k_loo = k_w4 + k_w2 + k_w1 + k_w3
# m = siggp.ModGP(X, y, k_w5, k_loo, k_g1, k_g2, Z, minibatch_size=Ns/1)
# m.kern1.fixed = True
# m.kern2.fixed = True
# m.kern3.fixed = True
# m.kern4.fixed = True
# m.likelihood.noise_var = noise_var
# m.likelihood.noise_var.fixed = True
#
# for i in range (winit, wfinish):
#     count = count + 1
#
#     X = Xtest[ i*Ns : (i+1)*Ns]
#     y = ytest[ i*Ns : (i+1)*Ns]
#     Z = X[::jump].copy()
#     #Z = np.vstack(( (X[::jump].copy()).reshape(-1,1) ,(X[-1].copy()).reshape(-1,1)  ))
#
#     m.X = X.copy()
#     m.Y = y.copy()
#     m.Z = Z.copy()
#
#     m.q_mu1._array = y[::jump].copy()
#     m.q_mu2._array = y[::jump].copy()
#     m.q_mu3._array = np.zeros(Z.shape)
#     m.q_mu4._array = np.zeros(Z.shape)
#
#     m.q_sqrt1._array = np.expand_dims(np.eye(Z.size), 2)
#     m.q_sqrt2._array = np.expand_dims(np.eye(Z.size), 2)
#     m.q_sqrt3._array = np.expand_dims(np.eye(Z.size), 2)
#     m.q_sqrt4._array = np.expand_dims(np.eye(Z.size), 2)
#
#     print('Analysing window number ', count, ' total number of windows to analyse ', Nw)
#     m.optimize(disp=1, maxiter = 200)
#
#     mu_f1, var_f1 = m.predict_f1(X)
#     mu_f2, var_f2 = m.predict_f2(X)
#     mu_g1, var_g1 = m.predict_g1(X)
#     mu_g2, var_g2 = m.predict_g2(X)
#
#     save_X[:,i-winit] = X.reshape(-1)
#     save_y[:,i-winit] = y.reshape(-1)
#     save_muf1[:,i-winit] = mu_f1.reshape(-1)
#     save_muf2[:,i-winit] = mu_f2.reshape(-1)
#     save_mug1[:,i-winit] = mu_g1.reshape(-1)
#     save_mug2[:,i-winit] = mu_g2.reshape(-1)
#     save_varf1[:,i-winit] = var_f1.reshape(-1)
#     save_varf2[:,i-winit] = var_f2.reshape(-1)
#     save_varg1[:,i-winit]= var_g1.reshape(-1)
#     save_varg2[:,i-winit]= var_g2.reshape(-1)
#     #ll = m.compute_log_likelihood()
#     #save_ll[i-winit,0] =  ll
#
# X = np.reshape(save_X, (-1,1), order = 'F')
# y = np.reshape(save_y, (-1,1), order = 'F')
# mu_f1 = np.reshape(save_muf1, (-1,1), order = 'F')
# mu_f2 = np.reshape(save_muf2, (-1,1), order = 'F')
# mu_g1 = np.reshape(save_mug1, (-1,1), order = 'F')
# mu_g2 = np.reshape(save_mug2, (-1,1), order = 'F')
# var_f1 = np.reshape(save_varf1, (-1,1), order = 'F')
# var_f2 = np.reshape(save_varf2, (-1,1), order = 'F')
# var_g1 = np.reshape(save_varg1, (-1,1), order = 'F')
# var_g2 = np.reshape(save_varg2, (-1,1), order = 'F')
#
# np.savez_compressed('../../results_FL/maps_5pitches', X = X,
#                               y = y,
#                               Xt1 = Xt1,
#                               yt1 = yt1,
#                               Xt2 = Xt2,
#                               yt2 = yt2,
#                               Xt3 = Xt3,
#                               yt3 = yt3,
#                               Xt4 = Xt4,
#                               yt4 = yt4,
#                               Xt5 = Xt5,
#                               yt5 = yt5,
#                               F   = F,
#                               S1 = S1,
#                               S2 = S2,
#                               S3 = S3,
#                               S4 = S4,
#                               S5 = S5,
#                               params = [par1, par2, par3, par4, par5],
#                               mu_f1 = mu_f1, var_f1 = var_f1,
#                               mu_f2 = mu_f2, var_f2 = var_f2,
#                               mu_g1 = mu_g1, var_g1 = var_g1,
#                               mu_g2 = mu_g2, var_g2 = var_g2)
#
# print 'Go!'
