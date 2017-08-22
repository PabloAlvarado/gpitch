'''This script is a demo of the new version for the modulated GP. The variable
ws defines the size (in number of samples) of the analysis window. choose ws=N
to analyze all data at once.'''
import numpy as np
from scipy import fftpack
import scipy as sp
from matplotlib import pyplot as plt
import tensorflow as tf
import GPflow
import time
import gpitch as gpi
import loogp
reload(loogp)
reload(gpi)


plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
plt.interactive(True)
plt.close('all')

# generate synthetic data
fs = 16e3  # sample frequency
N = 1600  # number of samples
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time
noise_var = 1.e-3

pitch1 = 440.00  # Hertz, A4 (La)
pitch2 = 554.37 # Hertz, D#5 (Do sharp)
pitch3 = 659.25  # Hertz, E5 (Mi)


kenv1 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
kenv2 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
kenv3 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)

kper1 = GPflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch1)
kper2 = GPflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch2)
kper3 = GPflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch3)

#kper2 = kper2_0 = kper2_1

Kenv1 = kenv1.compute_K_symm(x)
Kenv2 = kenv2.compute_K_symm(x)
Kenv3 = kenv3.compute_K_symm(x)
Kper1 = kper1.compute_K_symm(x)
Kper2 = kper2.compute_K_symm(x)
Kper3 = kper3.compute_K_symm(x)

np.random.seed(29)
f1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper1).reshape(-1, 1)
f2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper2).reshape(-1, 1)
f3 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper3).reshape(-1, 1)
f1 /= 3.*np.max(np.abs(f1))
f2 /= 3.*np.max(np.abs(f2))
f3 /= 3.*np.max(np.abs(f3))
f1 = f1 - f1.mean()
f2 = f2 - f2.mean()
f3 = f3 - f3.mean()
g1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv1).reshape(-1, 1)
g2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv2).reshape(-1, 1)
g3 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv3).reshape(-1, 1)
source1 = gpi.logistic(g1)*f1
source2 = gpi.logistic(g2)*f2
source3 = gpi.logistic(g3)*f3
mean = source1 + source2 + source3

y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)

s1, s2, s3 = [fftpack.fft(signal.reshape(-1,)) for signal in (source1, source2,
                                                              source3)]
T = 1. / fs
F = np.linspace(0., 0.5*fs, N/2)
S1 = 2.0/N * np.abs(s1[0:N/2])
S2 = 2.0/N * np.abs(s2[0:N/2])
S3 = 2.0/N * np.abs(s3[0:N/2])

plt.figure()
plt.plot(F, S1, '')
plt.plot(F, S2, '')
plt.plot(F, S3, '')


y1_ifft = np.fft.ifft(s1)
plt.plot(source1, '.')
plt.plot(y1_ifft)
aux0 = np.fft.ifft(np.abs(s1))
plt.figure()
plt.plot(aux0[0:800])


ker_test = 2.0*aux0[0:800]

#plt.figure()
#plt.plot(F, F1, '')
#plt.plot(F, F2, '')
#plt.plot(F, F3, '')


idx = np.argmax(S1)
a, b = idx - 25, idx + 25
if a < 0:
    a = 0
X, y = F[a: b,].reshape(-1,), S1[a: b,].reshape(-1,)

p0 = np.array([1.0, 0.1, 2*np.pi*F[idx]])
phat = sp.optimize.minimize(gpi.Lloss, p0, method='L-BFGS-B', args=(X, y),
                            tol = 1e-5, options={'disp': True})
pstar = phat.x
Xaux = np.linspace(X.min(), X.max(), 1000)
L = gpi.Lorentzian(pstar,Xaux)
plt.figure(), plt.xlim([X.min(), X.max()])
plt.plot(Xaux, L, '.', ms=8)
plt.plot(X, y, '.', ms=8)
print pstar
plt.figure()
plt.plot(X,y)

a1, b1, c1 = gpi.learnparams(X=F, S=S1, Nh=10)
Nh1 = a1.size
a2, b2, c2 = gpi.learnparams(X=F, S=S2, Nh=10)
Nh2 = a2.size
a3, b3, c3 = gpi.learnparams(X=F, S=S3, Nh=10)
Nh3 = a3.size

kper1_plot = gpi.MaternSM(x=x, s=a1, l=1./b1, f=2*np.pi*c1)

plt.figure()
plt.plot(aux0[0:800])
plt.figure()
plt.plot(kper1_plot)

# S2k = gpi.LorM(x=Faux, s=a2, l=1./b2, f=2*np.pi*c2 )
# S3k = gpi.LorM(x=Faux, s=a3, l=1./b3, f=2*np.pi*c3 )
#
# f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
# ax1.plot(F, S1, '.k')
# ax1.plot(Faux, S1k, 'g', lw=2)
# ax1.set_xlim([0, 3000])
# ax2.plot(F, S2, '.k')
# ax2.plot(Faux, S2k, 'g', lw=2)
# ax2.set_xlim([0, 3000])
# ax3.plot(F, S3, '.k')
# ax3.plot(Faux, S3k, 'g', lw=2)
# ax3.set_xlim([0, 3000])
#
#
# kper_msm_1 = gpi.ker_msm(s=a1, l=b1, f=c1, Nh=Nh1)
# kper_msm_2 = gpi.ker_msm(s=a2, l=b2, f=c2, Nh=Nh2)
# kper_msm_3 = gpi.ker_msm(s=a3, l=b3, f=c3, Nh=Nh3)
#
# kper1_plot = gpi.MaternSM(x=x, s=a1, l=1./b1, f=2*np.pi*c1)
# kper2_plot = gpi.MaternSM(x=x, s=a2, l=1./b2, f=2*np.pi*c2)
# kper3_plot = gpi.MaternSM(x=x, s=a3, l=1./b3, f=2*np.pi*c3)
#
# plt.figure(), plt.title('kernel 1')
# plt.plot(x, kper1_plot)
#
# plt.figure(), plt.title('kernel 2')
# plt.plot(x, kper2_plot)
#
# plt.figure(), plt.title('kernel 3')
# plt.plot(x, kper3_plot)
#
#
# ks1, ks2, ks3 = [fftpack.fft(i.reshape(-1,)) for i in (kper1_plot, kper2_plot,
#                                                               kper3_plot)]
# kS1 = 1./N * np.abs(ks1[0:N/2])
# kS2 = 1./N * np.abs(ks2[0:N/2])
# kS3 = 1./N * np.abs(ks3[0:N/2])
#
#
# plt.figure()
# plt.plot(F, kS1, '')
# plt.plot(F, S1, '')
#
# # plt.figure()
# # plt.plot(F, kS2, '')
# # plt.plot(F, S2, '')
# #
# # plt.figure()
# # plt.plot(F, kS3, '')
# # plt.plot(F, S3, '')
# #
#
# # #ttk2 = gpi.MaternSM(Xk, s=s_2, l=1./l_2, f=2*np.pi*f_2)
# #
# # #
# # # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
# # # ax1.plot(x, y)
# # # ax2.plot(x, source1)
# # # ax3.plot(x, source2)
# # # ax4.plot(x, source3)
# # # f.subplots_adjust(hspace=0)
# # #
# # # f, (ax2, ax3, ax4) = plt.subplots(3, sharex=True, sharey=True)
# # # ax2.plot(x, f1)
# # # ax3.plot(x, f2)
# # # ax4.plot(x, f3)
# # # f.subplots_adjust(hspace=0)
# # #
# # # f, (ax2, ax3, ax4) = plt.subplots(3, sharex=True, sharey=True)
# # # ax2.plot(x, gpi.logistic(g1))
# # # ax3.plot(x, gpi.logistic(g2))
# # # ax4.plot(x, gpi.logistic(g3))
# # # f.subplots_adjust(hspace=0)
# # #
# # # f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# # # ax1.plot(x, source1)
# # # ax2.plot(x, source2 + source3)
# # #
# # #
# # # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
# # # ax1.plot(x, source2 + source3)
# # # ax2.plot(x, source2)
# # # ax3.plot(x, source3)
# #
# #
# #
# # # split data into windows
# # ws = 400 # window size (samples)
# # #ws = N  # use all data at once (i.e. no windowing)
# # Nw = N/ws  # number of windows
# # x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]
# # y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]
# #
# # jump = 30  # initialize model
# # z = x_l[0][::jump].copy()
# #
# # kloo = kper_msm_2 + kper_msm_3
# # m = loogp.LooGP(x_l[0].copy(), y_l[0].copy(), [kper_msm_1, kloo], [kenv1, kenv2], z,
# #                 whiten=True)
# # m.likelihood.noise_var = noise_var
# # m.likelihood.noise_var.fixed = True
# # m.kern_f1.fixed = True
# # m.kern_f2.fixed = True
# # m.kern_g1.fixed = False
# # m.kern_g2.fixed = False
# #
# # qm1 = [np.zeros(z.shape) for i in range(0, Nw)]  # list to save predictions
# # qm2 = [np.zeros(z.shape) for i in range(0, Nw)]  # mean (qm) and variance (qv)
# # qm3 = [np.zeros(z.shape) for i in range(0, Nw)]
# # qm4 = [np.zeros(z.shape) for i in range(0, Nw)]
# # qv1 = [np.zeros(z.shape) for i in range(0, Nw)]
# # qv2 = [np.zeros(z.shape) for i in range(0, Nw)]
# # qv3 = [np.zeros(z.shape) for i in range(0, Nw)]
# # qv4 = [np.zeros(z.shape) for i in range(0, Nw)]
# #
# # maxiter = 500
# # start_time = time.time()
# # for i in range(Nw):
# #     m.X = x_l[i].copy()
# #     m.Y = y_l[i].copy()
# #     m.Z = x_l[i][::jump].copy()
# #
# #     m.q_mu1._array = np.zeros(z.shape)
# #     m.q_mu2._array = np.zeros(z.shape)
# #     m.q_mu3._array = np.zeros(z.shape)
# #     m.q_mu4._array = np.zeros(z.shape)
# #     m.q_sqrt1._array = np.expand_dims(np.eye(z.size), 2)
# #     m.q_sqrt2._array = np.expand_dims(np.eye(z.size), 2)
# #     m.q_sqrt3._array = np.expand_dims(np.eye(z.size), 2)
# #     m.q_sqrt4._array = np.expand_dims(np.eye(z.size), 2)
# #
# #     m.optimize(disp=1, maxiter=maxiter)
# #     qm1[i], qv1[i] = m.predict_f1(x_l[i])
# #     qm2[i], qv2[i] = m.predict_g1(x_l[i])
# #     qm3[i], qv3[i] = m.predict_f2(x_l[i])
# #     qm4[i], qv4[i] = m.predict_g2(x_l[i])
# #
# # print("--- %s seconds ---" % (time.time() - start_time))
# #
# # qm1 = np.asarray(qm1).reshape(-1, 1)
# # qm2 = np.asarray(qm2).reshape(-1, 1)
# # qm3 = np.asarray(qm3).reshape(-1, 1)
# # qm4 = np.asarray(qm4).reshape(-1, 1)
# #
# # qv1 = np.asarray(qv1).reshape(-1, 1)
# # qv2 = np.asarray(qv2).reshape(-1, 1)
# # qv3 = np.asarray(qv3).reshape(-1, 1)
# # qv4 = np.asarray(qv4).reshape(-1, 1)
# #
# # yhat = gpi.logistic(qm2)*qm1 + gpi.logistic(qm4)*qm3
# #
# # col = '#0172B2'
# # plt.figure(), plt.title('Mixture data and approximation')
# # plt.plot(x, y, '.k', mew=1)
# # plt.plot(x, yhat, color=col , lw=2)
# #
# # f, axarr = plt.subplots(2, sharex=True)
# # axarr[0].set_title('Latent quasi-periodic function 1 ')
# # axarr[0].plot(x, f1, '.k', mew=1)
# # axarr[0].plot(x, qm1, color=col, lw=2)
# # axarr[0].fill_between(x[:, 0], qm1[:, 0] - 2*np.sqrt(qv1[:, 0]),
# #                  qm1[:, 0] + 2*np.sqrt(qv1[:, 0]),
# #                  color=col, alpha=0.2)
# # axarr[1].set_title('Latent quasi-periodic function 2 ')
# # axarr[1].plot(x, f2, '.k', mew=1)
# # axarr[1].plot(x, qm3, color=col, lw=2)
# # axarr[1].twinx()
# # #axarr[1].fill_between(x[:, 0], qm3[:, 0] - 2*np.sqrt(qv3[:, 0]),
# # #                 qm3[:, 0] + 2*np.sqrt(qv3[:, 0]),
# # #                 color=col, alpha=0.2)
# #
# # f, axarr = plt.subplots(2, sharex=True)
# # axarr[0].set_title('Latent envelope 1 ')
# # axarr[0].plot(x[::5], gpi.logistic(g1[::5]), '.k', mew=1)
# # axarr[0].plot(x, gpi.logistic(qm2), 'g', lw=2)
# # axarr[0].fill_between(x[:, 0], gpi.logistic(qm2[:, 0] - 2*np.sqrt(qv2[:, 0])),
# #                   gpi.logistic(qm2[:, 0] + 2*np.sqrt(qv2[:, 0])),
# #                   color='green', alpha=0.2)
# # axarr[1].set_title('Latent envelope 2 ')
# # axarr[1].plot(x[::5], gpi.logistic(g2[::5]), '.k', mew=1)
# # axarr[1].plot(x, gpi.logistic(qm4), 'g', lw=2)
# # axarr[1].fill_between(x[:, 0], gpi.logistic(qm4[:, 0] - 2*np.sqrt(qv4[:, 0])),
# #                   gpi.logistic(qm4[:, 0] + 2*np.sqrt(qv4[:, 0])),
# #                   color='green', alpha=0.2)
# #
# # f, axarr = plt.subplots(2, sharex=True)
# # axarr[0].set_title('Latent source 1 (A4)')
# # axarr[0].plot(x, source1, '.k', mew=1)
# # axarr[0].plot(x, gpi.logistic(qm2)*qm1, color=col, lw=2)
# # axarr[1].set_title('Latent source 2 (E5)')
# # axarr[1].plot(x, source2 + source3 , '.k')
# # axarr[1].plot(x, gpi.logistic(qm4)*qm3, color=col, lw=2)
# #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
