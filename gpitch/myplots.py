import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../')
from gpitch.amtgp import logistic


def plot_results(mean_f, var_f, mean_g, var_g, x_plot, y):
    xla, xlb = 0.00, 0.025  # xlim plot
    mean_act = logistic(mean_g)
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 1, 1), plt.title('data, and approximation')
    plt.plot(x_plot, y, '.k')
    plt.twinx()
    plt.plot(x_plot, mean_act * mean_f, lw=2)
    #plt.xlim([xla, xlb])

    plt.subplot(3, 1, 2), plt.title('activation')
    plt.plot(x_plot, mean_act, 'C1', lw=2)
    plt.fill_between(x_plot, logistic(mean_g-2*np.sqrt(var_g)), logistic(mean_g+2*np.sqrt(var_g)), color='C1',
                     alpha=0.2)
    #plt.xlim([xla, xlb])

    plt.subplot(3, 1, 3), plt.title('component')
    plt.plot(x_plot, mean_f, 'C2', lw=2)
    plt.fill_between(x_plot, mean_f - 2 * np.sqrt(var_f), mean_f + 2 * np.sqrt(var_f), color='C2', alpha=0.2)
    #plt.xlim([xla, xlb])