import time, sys, os
sys.path.append('../../')
import matplotlib
if True: matplotlib.use('agg') # define if running code on server (True)
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow, gpitch
from gpitch.amtgp import logistic


np.random.seed(23)
gpitch.amtgp.init_settings(visible_device=sys.argv[1], interactive=True) #  confi gpu usage, plot
fs = 16e3  # generate synthetic data
N = 1600  # number of samples
x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time
noise_var = 1.e-3  # noise variance
pitch1 = 440.00  # Hertz, A4 (La)
pitch2 = 659.25  # Hertz, E5 (Mi)
kenv1 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.01, variance=10.)
kenv2 = gpflow.kernels.Matern32(input_dim=1, lengthscales=0.005, variance=10.)
kper1 = gpflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch1)
kper2 = gpflow.kernels.PeriodicKernel(input_dim=1, lengthscales=0.25,
                                      variance=np.sqrt(0.5), period=1./pitch2)
Kenv1 = kenv1.compute_K_symm(x)
Kenv2 = kenv2.compute_K_symm(x)
Kper1 = kper1.compute_K_symm(x)
Kper2 = kper2.compute_K_symm(x)
f1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper1).reshape(-1, 1)
f2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kper2).reshape(-1, 1)
f1 /= np.max(np.abs(f1))
f2 /= np.max(np.abs(f2))
g1 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv1).reshape(-1, 1)
g2 = np.random.multivariate_normal(np.zeros(x.shape[0]), Kenv2).reshape(-1, 1)
source1 = gpitch.amtgp.logistic(g1)*f1
source2 = gpitch.amtgp.logistic(g2)*f2
mean = source1 + source2
y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)

ws = N  # use all data at once (i.e. no windowing)
Nw = N/ws  # number of windows
x_l = [x[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]  # split data into windows
y_l = [y[i*ws:(i+1)*ws].copy() for i in range(0, Nw)]
jump = 10  # initialize model
z = x_l[0][::jump].copy()
m = gpitch.loogp.LooGP(x_l[0].copy(), y_l[0].copy(), [kper1, kper2], [kenv1, kenv2], z, whiten=True)
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
maxiter = 500
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
qm1 = np.asarray(qm1).reshape(-1, )
qm2 = np.asarray(qm2).reshape(-1, )
qm3 = np.asarray(qm3).reshape(-1, )
qm4 = np.asarray(qm4).reshape(-1, )
qv1 = np.asarray(qv1).reshape(-1, )
qv2 = np.asarray(qv2).reshape(-1, )
qv3 = np.asarray(qv3).reshape(-1, )
qv4 = np.asarray(qv4).reshape(-1, )
yhat = logistic(qm2)*qm1 + logistic(qm4)*qm3
x = x.reshape(-1, )
y = y.reshape(-1, )

def plot_results():
    nrows = 4
    ncols = 2
    plt.figure(figsize=(ncols*18, nrows*6))
    plt.subplot(nrows, ncols, (1, 2))
    plt.title('data and prediction')
    plt.plot(x, y, '.k', mew=1)
    plt.plot(x, yhat, lw=2)

    plt.subplot(nrows, ncols, 3)
    plt.title('component 1')
    plt.plot(x, f1, '.k', mew=1)
    plt.plot(x, qm1, color='C0', lw=2)
    plt.fill_between(x, qm1-2*np.sqrt(qv1), qm1+2*np.sqrt(qv1), color='C0', alpha=0.2)

    plt.subplot(nrows, ncols, 4)
    plt.title('component 2')
    plt.plot(x, f2, '.k', mew=1)
    plt.plot(x, qm3, color='C0', lw=2)
    plt.fill_between(x, qm3-2*np.sqrt(qv3), qm3+2*np.sqrt(qv3), color='C0', alpha=0.2)

    plt.subplot(nrows, ncols, 5)
    plt.title('activation 1')
    plt.plot(x[::5], logistic(g1[::5]), '.k', mew=1)
    plt.plot(x, logistic(qm2), 'g', lw=2)
    plt.fill_between(x, logistic(qm2-2*np.sqrt(qv2)), logistic(qm2+2*np.sqrt(qv2)),
                     color='g', alpha=0.2)

    plt.subplot(nrows, ncols, 6)
    plt.title('activation 2')
    plt.plot(x[::5], logistic(g2[::5]), '.k', mew=1)
    plt.plot(x, logistic(qm4), 'g', lw=2)
    plt.fill_between(x, logistic(qm4-2*np.sqrt(qv4)), logistic(qm4+2*np.sqrt(qv4)),
                     color='g', alpha=0.2)

    plt.subplot(nrows, ncols, 7)
    plt.title('source 1')
    plt.plot(x, source1, '.k', mew=1)
    plt.plot(x, logistic(qm2)*qm1, lw=2)

    plt.subplot(nrows, ncols, 8)
    plt.title('source 2')
    plt.plot(x, source2, '.k', mew=1)
    plt.plot(x, logistic(qm4)*qm3, lw=2)

plot_results()
plt.tight_layout()
plt.savefig('../../../results/figures/demos/demo_loogp_toy_old.png')






























#
