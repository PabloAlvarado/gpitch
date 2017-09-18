import numpy as np
import scipy as sp
from scipy import fftpack
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import GPflow
import time
import gpitch as gpi
import loogp
import sounddevice as sod #reproduce audio from numpy arrays
import soundfile  # package to load wav files
reload(loogp)
reload(gpi)


plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
plt.interactive(True)
plt.close('all')

data, fs = soundfile.read('syn_data.wav')
data = 0.5*np.sum(data, 1).reshape(-1,1)
#plt.figure()
#plt.plot(data)

y_c = data[0:3*44100].copy()
y_e = data[176600:176600 + 3*44100].copy()
y_g = data[353250:353250 + 3*44100].copy()
N = np.size(y_c)
x = np.linspace(0, (N-1.)/fs, N)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, y_c)
ax2.plot(x, y_e)
ax3.plot(x, y_g)

Yc = fftpack.fft(y_c.reshape(-1,))
Ye = fftpack.fft(y_e.reshape(-1,))
Yg = fftpack.fft(y_g.reshape(-1,))
F = np.linspace(0., 0.5*fs, N/2)
Sc =  2.0/N*np.abs(Yc[0:N/2])
Se =  2.0/N*np.abs(Ye[0:N/2])
Sg =  2.0/N*np.abs(Yg[0:N/2])
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.plot(F, Sc)
ax2.plot(F, Se)
ax3.plot(F, Sg)
ax3.set_xlim([0, 8e3])


ker_c_hat =  fftpack.ifftshift(fftpack.ifft(np.abs(Yc)))
ker_e_hat =  fftpack.ifftshift(fftpack.ifft(np.abs(Ye)))
ker_g_hat =  fftpack.ifftshift(fftpack.ifft(np.abs(Yg)))
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.plot(x- x[-1]/2, ker_c_hat)
ax2.plot(x- x[-1]/2, ker_e_hat)
ax3.plot(x- x[-1]/2, ker_g_hat)

# idx = np.argmax(S)
# a, b = idx - 25, idx + 25
# if a < 0:
#     a = 0
# X, y = F[a: b,].reshape(-1,), S[a: b,].reshape(-1,)
#
# plt.figure()
# plt.plot(X, y, '.')
#
# p0 = np.array([1.0, 0.1, 2*np.pi*F[idx]])
# phat = sp.optimize.minimize(gpi.Lloss, p0, method='L-BFGS-B', args=(X, y),
#                             tol = 1e-16, options={'disp': True})
# pstar = phat.x
# Xaux = np.linspace(X.min(), X.max(), 1000)
# L = gpi.Lorentzian(pstar,Xaux)
# plt.figure(), plt.xlim([X.min(), X.max()])
# plt.plot(Xaux, L, '.', ms=8)
# plt.plot(X, y, '.', ms=8)
# print pstar
# plt.figure()
# plt.plot(X,y)
#pstar = np.asarray(pstar).reshape(-1,1)
#kper1_plot = gpi.MaternSM(x=x, s=pstar[0], l=1./pstar[1], f=2*np.pi*pstar[2])
#plt.plot(kper1_plot)
#plt.ylim([-1, 1])

a1, b1, c1 = gpi.learnparams(X=F, S=Sc, Nh=10)
a2, b2, c2 = gpi.learnparams(X=F, S=Se, Nh=10)
a3, b3, c3 = gpi.learnparams(X=F, S=Sg, Nh=10)

Sker_c = gpi.LorM(x=F, s=a1, l=1./b1, f=2*np.pi*c1)
Sker_e = gpi.LorM(x=F, s=a2, l=1./b2, f=2*np.pi*c2)
Sker_g = gpi.LorM(x=F, s=a3, l=1./b3, f=2*np.pi*c3)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.plot(F, Sc)
ax1.plot(F, Sker_c)
ax2.plot(F, Se)
ax2.plot(F, Sker_e)
ax3.plot(F, Sg)
ax3.plot(F, Sker_g)


ker_c = gpi.MaternSM(x=x -x[-1]/2., s=a1, l=1./b1, f=2*np.pi*c1)
ker_e = gpi.MaternSM(x=x -x[-1]/2., s=a2, l=1./b2, f=2*np.pi*c2)
ker_g = gpi.MaternSM(x=x -x[-1]/2., s=a3, l=1./b3, f=2*np.pi*c3)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x- x[-1]/2., ker_c_hat)
ax1.plot(x -x[-1]/2., ker_c)
ax2.plot(x- x[-1]/2., ker_e_hat)
ax2.plot(x -x[-1]/2., ker_e)
ax3.plot(x- x[-1]/2., ker_g_hat)
ax3.plot(x -x[-1]/2., ker_g)

y_test = data[529400:-1].reshape(-1,1)
Nt = y_test.size
x_test = np.linspace(0,(Nt-1)/fs, Nt ).reshape(-1,1)
plt.figure()
plt.plot(x_test, y_test)

y_t = y_test[0:1600]
x_t = x_test[0:1600]
plt.figure()
plt.plot(x_t, y_t)





kper_msm_1 = gpi.ker_msm(s=a1, l=b1, f=c1, Nh=a1.size)
kper_msm_2 = gpi.ker_msm(s=a2, l=b2, f=c2, Nh=a2.size)
kper_msm_3 = gpi.ker_msm(s=a3, l=b3, f=c3, Nh=a3.size)

kenv1 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
kenv2 = GPflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)

# split data into windows
ws = 1600 # window size (samples)
#ws = N  # use all data at once (i.e. no windowing)
Nw = y_t.size/ws  # number of windows
x_l = [x_t[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]
y_l = [y_t[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]

noise_var = 1.e-3
jump = 20  # initialize model
z = x_l[0][::jump].copy()

kloo = kper_msm_2
m = loogp.LooGP(x_l[0].copy(), y_l[0].copy(), [kper_msm_1, kloo], [kenv1, kenv2], z,
                whiten=True)
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True
m.kern_f1.fixed = True
m.kern_f2.fixed = True
m.kern_g1.fixed = True
m.kern_g2.fixed = True

qm1 = [np.zeros(z.shape) for i in range(0, Nw)]  # list to save predictions
qm2 = [np.zeros(z.shape) for i in range(0, Nw)]  # mean (qm) and variance (qv)
qm3 = [np.zeros(z.shape) for i in range(0, Nw)]
qm4 = [np.zeros(z.shape) for i in range(0, Nw)]
qv1 = [np.zeros(z.shape) for i in range(0, Nw)]
qv2 = [np.zeros(z.shape) for i in range(0, Nw)]
qv3 = [np.zeros(z.shape) for i in range(0, Nw)]
qv4 = [np.zeros(z.shape) for i in range(0, Nw)]

maxiter = 250
start_time = time.time()
for i in range(Nw):
    m.X = x_l[i].copy()
    m.Y = y_l[i].copy()
    m.Z = x_l[i][::jump].copy()

    m.q_mu1._array = np.zeros(z.shape)
    m.q_mu2._array = np.zeros(z.shape)
    m.q_mu3._array = np.zeros(z.shape)
    m.q_mu4._array = np.zeros(z.shape)
    m.q_sqrt1._array = np.expand_dims(np.eye(z.size), 2)
    m.q_sqrt2._array = np.expand_dims(np.eye(z.size), 2)
    m.q_sqrt3._array = np.expand_dims(np.eye(z.size), 2)
    m.q_sqrt4._array = np.expand_dims(np.eye(z.size), 2)

    m.optimize(disp=1, maxiter=maxiter)
    qm1[i], qv1[i] = m.predict_f1(x_l[i])
    qm2[i], qv2[i] = m.predict_g1(x_l[i])
    qm3[i], qv3[i] = m.predict_f2(x_l[i])
    qm4[i], qv4[i] = m.predict_g2(x_l[i])

print("--- %s seconds ---" % (time.time() - start_time))

qm1 = np.asarray(qm1).reshape(-1, 1)
qm2 = np.asarray(qm2).reshape(-1, 1)
qm3 = np.asarray(qm3).reshape(-1, 1)
qm4 = np.asarray(qm4).reshape(-1, 1)

qv1 = np.asarray(qv1).reshape(-1, 1)
qv2 = np.asarray(qv2).reshape(-1, 1)
qv3 = np.asarray(qv3).reshape(-1, 1)
qv4 = np.asarray(qv4).reshape(-1, 1)

yhat = gpi.logistic(qm2)*qm1 + gpi.logistic(qm4)*qm3

col = '#0172B2'
plt.figure(), plt.title('Mixture data and approximation')
plt.plot(x_t, y_t, '.k', mew=1)
plt.plot(x_t, yhat, color=col , lw=2)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Latent source 1')
axarr[0].plot(x_t, gpi.logistic(qm2)*qm1, color=col, lw=2)
axarr[1].set_title('Latent source 2')
axarr[1].plot(x_t, gpi.logistic(qm4)*qm3, color=col, lw=2)

























#
