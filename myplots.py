import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../')
from gpitch.amtgp import logistic


def plot_results(mean_f, var_f, mean_g, var_g, x_plot, y, z, xlim):
    mean_act = logistic(mean_g)

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1), plt.title('data, and approximation')
    plt.plot(x_plot, mean_act * mean_f, lw=2)
    plt.plot(x_plot, y, '.k')
    plt.plot(z, np.zeros(z.shape), 'r|', mew=4)
    plt.xlim(xlim)

    plt.subplot(3, 1, 2), plt.title('activation')
    plt.plot(x_plot, mean_act, 'C1', lw=2)
    plt.fill_between(x_plot, logistic(mean_g-2*np.sqrt(var_g)), logistic(mean_g+2*np.sqrt(var_g)), color='C1',
                     alpha=0.2)
    plt.plot(z, np.zeros(z.shape), 'r|', mew=4)
    plt.xlim(xlim)

    plt.subplot(3, 1, 3), plt.title('component')
    plt.plot(x_plot, mean_f, 'C2', lw=2)
    plt.fill_between(x_plot, mean_f - 2 * np.sqrt(var_f), mean_f + 2 * np.sqrt(var_f), color='C2', alpha=0.2)
    plt.plot(z, np.zeros(z.shape), 'r|', mew=4)
    plt.xlim(xlim)


def plot_loo(mean_f, var_f, mean_g, var_g, x_plot, y, z, xlim):
    mean_f1, mean_f2 = mean_f[0], mean_f[1]
    mean_g1, mean_g2 = mean_g[0], mean_g[1]
    var_f1, var_f2 = var_f[0], var_f[1]
    var_g1, var_g2 = var_g[0], var_g[1]
    x_plot = x_plot.reshape(-1,)
    y_plot = y.reshape(-1,)

    ncol, nrow = 5, 2
    plt.figure(figsize=(16, 12))

    plt.subplot(ncol, nrow, 1)
    plt.plot(x_plot, logistic(mean_g1)*mean_f1)
    plt.ylim([-1, 1])

    plt.subplot(ncol, nrow, 2)
    plt.plot(x_plot, logistic(mean_g2)*mean_f2)
    plt.ylim([-1, 1])

    plt.subplot(ncol, nrow, 3)
    plt.plot(x_plot, mean_f1)

    plt.subplot(ncol, nrow, 4)
    plt.plot(x_plot, mean_f2)

    plt.subplot(ncol, nrow, 5)
    plt.plot(x_plot, logistic(mean_g1), 'C1')
    plt.fill_between(x_plot, logistic(mean_g1-2*np.sqrt(var_g1)),
                     logistic(mean_g1+2*np.sqrt(var_g1)), color='C1', alpha=0.2)

    plt.subplot(ncol, nrow,6)
    plt.plot(x_plot, logistic(mean_g2), 'C1')
    plt.fill_between(x_plot, logistic(mean_g2-2*np.sqrt(var_g2)),
                     logistic(mean_g2+2*np.sqrt(var_g2)), color='C1', alpha=0.2)

    plt.subplot(ncol, nrow, (7,8))
    plt.plot(x_plot, y, '.k')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1 + logistic(mean_g2)*mean_f2, 'C0', lw=2)
    plt.plot(z, -np.ones((z.shape)), '|r')
