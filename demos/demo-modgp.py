from gpitch.matern12_spectral_mixture import MercerMatern12sm
from gpitch.myplots import plot_predict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpitch
import gpflow


def per_fun(xin, npartials, freq):
    """Function to generate sum os sines"""
    f = np.zeros(xin.shape)
    for i in range(npartials):
        f += np.sin(2 * np.pi * xin * (i+1) * freq)
    return f/np.max(np.abs(f))


# generate data
n = 16000  # number of samples
fs = 16000  # sample frequency
x = np.linspace(0., (n-1.)/fs, n).reshape(-1, 1)
component = per_fun(xin=x, npartials=3, freq=15.)
envelope = np.exp(-25 * (x - 0.33) ** 2) + np.exp(-75 * (x - 0.66) ** 2)
envelope /= np.max(np.abs(envelope))
noise_var = 0.000001
y = component * envelope + np.sqrt(noise_var) * np.random.randn(component.size, 1)

#  use maxima as inducing points
z, u = gpitch.init_liv(x=x, y=y, win_size=31, thres=0.05, dec=1)

# init kernels
kact = gpflow.kernels.Matern32(input_dim=1, lengthscales=1.0, variance=1.0)
enr = np.array([1., 1., 1.])
frq = np.array([15., 30., 45.])
kcom = MercerMatern12sm(input_dim=1, energy=enr, frequency=frq)
kern = [[kact], [kcom]]

# init model
m = gpitch.pdgp.Pdgp(x=x.copy(), y=y.copy(), z=z, kern=kern, minibatch_size=100)
m.za.fixed = True
m.zc.fixed = True

# optimization
method = tf.train.AdamOptimizer(learning_rate=0.005)
m.optimize(method=method, maxiter=2500)

# predict
xtest = x[::4].copy()
mu_a, var_a, mu_c, var_c, m_src = m.predict_act_n_com(xtest)

# plot data
plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(x, y, 'k--')
plt.plot(xtest, m_src[0])
plt.plot(z[0][0], u, 'o', mfc="none")
plt.legend(['data', 'prediction', 'maxima data (ind. points)'])

plt.subplot(3, 1, 2)
plt.plot(x, envelope, 'k--')
plt.legend(['envelope'], loc=1)
plt.twinx()
plot_predict(x=xtest, mean=mu_a[0], var=var_a[0], z=m.za[0].value, latent=True)
plt.legend(['prediction', 'inducing points'], loc=2)

plt.subplot(3, 1, 3)
plt.plot(x, component, 'k--')
plt.legend(['component'], loc=1)
plt.twinx()
plot_predict(x=xtest, mean=mu_c[0], var=var_c[0], z=m.zc[0].value)
plt.legend(['prediction', 'inducing points'], loc=2)
plt.savefig("demo-modgp.png")
plt.show()
