import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../')
from gpitch import logistic, gaussfunc
from scipy.fftpack import fft, ifft, ifftshift


def plot_results(mean_f, var_f, mean_g, var_g, x_plot, y, z, xlim):
    mean_act = logistic(mean_g)
    plt.figure()
    plt.subplot(1, 4, 1), plt.title('data')
    plt.plot(x_plot, y)
    plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 2), plt.title('data approx')
    plt.plot(x_plot, mean_act * mean_f, lw=2)
    plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 3), plt.title('activation')
    plt.plot(x_plot, mean_act, 'C0', lw=2)
    plt.fill_between(x_plot, logistic(mean_g-2*np.sqrt(var_g)), logistic(mean_g+2*np.sqrt(var_g)), color='C0',
                     alpha=0.2)
    plt.plot(z, np.zeros(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 4), plt.title('component')
    plt.plot(x_plot, mean_f, 'C0', lw=2)
    plt.fill_between(x_plot, mean_f - 2 * np.sqrt(var_f), mean_f + 2 * np.sqrt(var_f), color='C0', alpha=0.2)
    plt.plot(z, np.zeros(z.shape), 'k|', mew=1)
    plt.xlim(xlim)


def plot_results_2(mean_f, var_f, mean_g, var_g, x_plot, y, za, zc, xlim):
    mean_act = logistic(mean_g)
    plt.figure()
    plt.subplot(1, 4, 1), plt.title('data')
    plt.plot(x_plot, y)
    # plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 2), plt.title('data approx')
    plt.plot(x_plot, mean_act * mean_f, lw=2)
    # plt.plot(z, -np.ones(z.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 3), plt.title('activation')
    plt.plot(x_plot, mean_act, 'C0', lw=2)
    plt.fill_between(x_plot, logistic(mean_g-2*np.sqrt(var_g)), logistic(mean_g+2*np.sqrt(var_g)), color='C0',
                     alpha=0.2)
    plt.plot(za, np.zeros(za.shape), 'k|', mew=1)
    plt.xlim(xlim)

    plt.subplot(1, 4, 4), plt.title('component')
    plt.plot(x_plot, mean_f, 'C0', lw=2)
    plt.fill_between(x_plot, mean_f - 2 * np.sqrt(var_f), mean_f + 2 * np.sqrt(var_f), color='C0', alpha=0.2)
    plt.plot(zc, np.zeros(zc.shape), 'k|', mew=1)
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
    plt.fill_between(x_plot, mean_f1 - 2*np.sqrt(var_f1),
                             mean_f1 + 2*np.sqrt(var_f1), color='C0', alpha=0.2)

    plt.subplot(ncol, nrow, 4)
    plt.plot(x_plot, mean_f2)
    plt.fill_between(x_plot, mean_f2 - 2*np.sqrt(var_f2),
                             mean_f2 + 2*np.sqrt(var_f2), color='C0', alpha=0.2)

    plt.subplot(ncol, nrow, 5)
    plt.plot(x_plot, logistic(mean_g1), 'C0')
    plt.fill_between(x_plot, logistic(mean_g1-2*np.sqrt(var_g1)),
                     logistic(mean_g1+2*np.sqrt(var_g1)), color='C0', alpha=0.2)

    plt.subplot(ncol, nrow,6)
    plt.plot(x_plot, logistic(mean_g2), 'C0')
    plt.fill_between(x_plot, logistic(mean_g2-2*np.sqrt(var_g2)),
                     logistic(mean_g2+2*np.sqrt(var_g2)), color='C0', alpha=0.2)

    plt.subplot(ncol, nrow, (7,8))
    plt.plot(x_plot, y, '.k')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1 + logistic(mean_g2)*mean_f2, 'C0', lw=2)
    plt.plot(z, -np.ones((z.shape)), '|r')


def plot_spec(iparam):
    """
    iparam[0] : F_star
    iparam[1] : S_star,
    iparam[2] : F,
    iparam[3] : Ss,
    iparam[4] : thres
    """
    F_star = iparam[0]
    S_star = iparam[1]
    F = iparam[2]
    S = iparam[3]
    plt.plot(F, S/np.max(S))
    plt.plot(F_star, S_star/np.max(S_star), 'ok', mew=2, mfc='none')
    plt.legend(['Smoothed spectral density of data', 'location of harmonics found for initialization'])
    plt.xlim([0, 8000])

def plot_kern_sd(model, x, iparam, scaled=True):
    """plot spectral density of kernel"""
    F = iparam[2] #  frequency vector
    S = iparam[3] #  spectral density data
    k_plot_model = model.kern_com.compute_K(x, np.asarray(0.).reshape(-1,1))
    N = x.size
    Yk1 = fft(k_plot_model.reshape(-1,)) #  FFT data
    Sk1 =  2./N * np.abs(Yk1[0:N//2]) #  spectral density data
    plt.plot(F, S / np.max(np.abs(S)), 'r', lw=2)
    plt.plot(F, Sk1 / np.max(np.abs(Sk1)), lw=2)
    plt.legend([' Spectral density component kernel', 'Spectral density data' ])
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 8000])



def plot_loo3(mean_f, var_f, mean_g, var_g, x_plot, y, z, xlim):
    mean_f1, mean_f2 = mean_f[0], mean_f[1]
    mean_g1, mean_g2 = mean_g[0], mean_g[1]
    var_f1, var_f2 = var_f[0], var_f[1]
    var_g1, var_g2 = var_g[0], var_g[1]
    x_plot = x_plot.reshape(-1,)
    y_plot = y.reshape(-1,)

    ncol, nrow = 5, 2
    plt.figure(figsize=(16, 10))
    
    plt.subplot(ncol, nrow, 1)#, plt.title('activation 1')
    plt.plot(x_plot, logistic(mean_g1), 'C0')
    plt.fill_between(x_plot, logistic(mean_g1-2*np.sqrt(var_g1)),
                     logistic(mean_g1+2*np.sqrt(var_g1)), color='C0', alpha=0.2)
    plt.plot(z[0], np.zeros((z[0].shape)), '|C3', mew=2)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 2)#, plt.title('activation 2')
    plt.plot(x_plot, logistic(mean_g2), 'C0')
    plt.fill_between(x_plot, logistic(mean_g2-2*np.sqrt(var_g2)),
                     logistic(mean_g2+2*np.sqrt(var_g2)), color='C0', alpha=0.2)
    plt.plot(z[2], np.zeros((z[2].shape)), '|C3', mew=2)
    plt.xlim(xlim)


    plt.subplot(ncol, nrow, 3)#, plt.title('component 1')
    plt.plot(x_plot, mean_f1, 'C0')
    plt.fill_between(x_plot, mean_f1 - 2*np.sqrt(var_f1),
                             mean_f1 + 2*np.sqrt(var_f1), color='C0', alpha=0.2)
    plt.plot(z[1], np.zeros((z[1].shape)), '|C3', mew=2)
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 4)
    plt.plot(x_plot, mean_f2, 'C0')#, plt.title('component 2')
    plt.fill_between(x_plot, mean_f2 - 2*np.sqrt(var_f2),
                             mean_f2 + 2*np.sqrt(var_f2), color='C0', alpha=0.2)
    plt.plot(z[3], np.zeros((z[3].shape)), '|C3', mew=2)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 5)#, plt.title('source 1')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 6)#, plt.title('source 2')
    plt.plot(x_plot, logistic(mean_g2)*mean_f2)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, (7))#, plt.title('data')
    plt.plot(x_plot, y, 'C0')
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, (8))#, plt.title('approximation')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1 + logistic(mean_g2)*mean_f2, 'C0')
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    plt.suptitle('Results')


def plot_ssgp(m, mean_f, var_f, mean_g, var_g, x_plot, y, title='results'):
    mean_f1, mean_f2, mean_f3 = mean_f[0].reshape(-1,), mean_f[1].reshape(-1,), mean_f[2].reshape(-1,)
    mean_g1, mean_g2, mean_g3 = mean_g[0].reshape(-1,), mean_g[1].reshape(-1,), mean_g[2].reshape(-1,)
    var_f1, var_f2, var_f3 = var_f[0].reshape(-1,), var_f[1].reshape(-1,), var_f[2].reshape(-1,)
    var_g1, var_g2, var_g3 = var_g[0].reshape(-1,), var_g[1].reshape(-1,), var_g[2].reshape(-1,)
    x_plot = x_plot.reshape(-1,)
    y_plot = y.reshape(-1,)
    xlim = [x_plot[0], x_plot[-1]]
    mew=1

    z = [m.Za1.value, m.Zc1.value, m.Za2.value, m.Zc2.value, m.Za3.value, m.Zc3.value]
    ncol, nrow = 4, 3
    plt.figure(figsize=(16, 10))
    
    plt.subplot(ncol, nrow, 1)#, plt.title('activation 1')
    plt.plot(x_plot, logistic(mean_g1), 'C0')
    plt.fill_between(x_plot, logistic(mean_g1-2*np.sqrt(var_g1)),
                     logistic(mean_g1+2*np.sqrt(var_g1)), color='C0', alpha=0.2)
    plt.twinx()
    plt.plot(x_plot, mean_g1, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g1-2*np.sqrt(var_g1), mean_g1+2*np.sqrt(var_g1), color='C2', alpha=0.1)
    
    plt.plot(z[0], np.zeros((z[0].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 2)#, plt.title('activation 2')
    plt.plot(x_plot, logistic(mean_g2), 'C0')
    plt.fill_between(x_plot, logistic(mean_g2-2*np.sqrt(var_g2)),
                     logistic(mean_g2+2*np.sqrt(var_g2)), color='C0', alpha=0.2)
    plt.twinx()
    plt.plot(x_plot, mean_g2, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g2-2*np.sqrt(var_g2), mean_g2+2*np.sqrt(var_g2), color='C2', alpha=0.1)
    
    plt.plot(z[2], np.zeros((z[2].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 3)#, plt.title('activation 3')
    plt.plot(x_plot, logistic(mean_g3), 'C0')
    plt.fill_between(x_plot, logistic(mean_g3-2*np.sqrt(var_g3)),
                     logistic(mean_g3+2*np.sqrt(var_g3)), color='C0', alpha=0.2)
    plt.twinx()
    plt.plot(x_plot, mean_g3, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g3-2*np.sqrt(var_g3), mean_g3+2*np.sqrt(var_g3), color='C2', alpha=0.1)
    
    plt.plot(z[4], np.zeros((z[4].shape)), '|C3', mew=mew)
    plt.xlim(xlim)


    plt.subplot(ncol, nrow, 4)#, plt.title('component 1')
    plt.plot(x_plot, mean_f1, 'C0')
    plt.fill_between(x_plot, mean_f1 - 2*np.sqrt(var_f1),
                             mean_f1 + 2*np.sqrt(var_f1), color='C0', alpha=0.2)
    plt.plot(z[1], np.zeros((z[1].shape)), '|C3', mew=mew)
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 5)
    plt.plot(x_plot, mean_f2, 'C0')#, plt.title('component 2')
    plt.fill_between(x_plot, mean_f2 - 2*np.sqrt(var_f2),
                             mean_f2 + 2*np.sqrt(var_f2), color='C0', alpha=0.2)
    plt.plot(z[3], np.zeros((z[3].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 6)
    plt.plot(x_plot, mean_f3, 'C0')#, plt.title('component 2')
    plt.fill_between(x_plot, mean_f3 - 2*np.sqrt(var_f3),
                             mean_f3 + 2*np.sqrt(var_f3), color='C0', alpha=0.2)
    plt.plot(z[5], np.zeros((z[5].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 7)#, plt.title('source 1')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 8)#, plt.title('source 2')
    plt.plot(x_plot, logistic(mean_g2)*mean_f2)
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 9)#, plt.title('source 2')
    plt.plot(x_plot, logistic(mean_g3)*mean_f3)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, (10,12))#, plt.title('data')
    plt.plot(x_plot, y, 'C0')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1 + logistic(mean_g2)*mean_f2 + logistic(mean_g3)*mean_f3, 'k')
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    plt.suptitle(title)
    
    
    
    
    
def plot_ssgp_gauss(m, mean_f, var_f, mean_g, var_g, x_plot, y, title='results'):
    mean_f1, mean_f2, mean_f3 = mean_f[0].reshape(-1,), mean_f[1].reshape(-1,), mean_f[2].reshape(-1,)
    mean_g1, mean_g2, mean_g3 = mean_g[0].reshape(-1,), mean_g[1].reshape(-1,), mean_g[2].reshape(-1,)
    var_f1, var_f2, var_f3 = var_f[0].reshape(-1,), var_f[1].reshape(-1,), var_f[2].reshape(-1,)
    var_g1, var_g2, var_g3 = var_g[0].reshape(-1,), var_g[1].reshape(-1,), var_g[2].reshape(-1,)
    x_plot = x_plot.reshape(-1,)
    y_plot = y.reshape(-1,)
    xlim = [x_plot[0], x_plot[-1]]
    mew=1

    z = [m.Za1.value, m.Zc1.value, m.Za2.value, m.Zc2.value, m.Za3.value, m.Zc3.value]
    ncol, nrow = 4, 3
    plt.figure(figsize=(16, 10))
    
    plt.subplot(ncol, nrow, 1)#, plt.title('activation 1')
    plt.plot(x_plot, gaussfunc(mean_g1), 'C0')
    plt.fill_between(x_plot, gaussfunc(mean_g1-2*np.sqrt(var_g1)),
                     gaussfunc(mean_g1+2*np.sqrt(var_g1)), color='C0', alpha=0.2)
    #plt.ylim(-0.1, 1.1)
    plt.twinx()
    plt.plot(x_plot, mean_g1, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g1-2*np.sqrt(var_g1), mean_g1+2*np.sqrt(var_g1), color='C2', alpha=0.1)
    
    plt.plot(z[0], np.zeros((z[0].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 2)#, plt.title('activation 2')
    plt.plot(x_plot, gaussfunc(mean_g2), 'C0')
    plt.fill_between(x_plot, gaussfunc(mean_g2-2*np.sqrt(var_g2)),
                     gaussfunc(mean_g2+2*np.sqrt(var_g2)), color='C0', alpha=0.2)
    #plt.ylim(-0.1, 1.1)
    plt.twinx()
    plt.plot(x_plot, mean_g2, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g2-2*np.sqrt(var_g2), mean_g2+2*np.sqrt(var_g2), color='C2', alpha=0.1)
    
    plt.plot(z[2], np.zeros((z[2].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 3)#, plt.title('activation 3')
    plt.plot(x_plot, gaussfunc(mean_g3), 'C0')
    plt.fill_between(x_plot, gaussfunc(mean_g3-2*np.sqrt(var_g3)),
                     gaussfunc(mean_g3+2*np.sqrt(var_g3)), color='C0', alpha=0.2)
    #plt.ylim(-0.1, 1.1)
    plt.twinx()
    plt.plot(x_plot, mean_g3, 'C2', alpha=0.5)
    plt.fill_between(x_plot, mean_g3-2*np.sqrt(var_g3), mean_g3+2*np.sqrt(var_g3), color='C2', alpha=0.1)
    plt.plot(z[4], np.zeros((z[4].shape)), '|C3', mew=mew)
    
    plt.xlim(xlim)


    plt.subplot(ncol, nrow, 4)#, plt.title('component 1')
    plt.plot(x_plot, mean_f1, 'C0')
    plt.fill_between(x_plot, mean_f1 - 2*np.sqrt(var_f1),
                             mean_f1 + 2*np.sqrt(var_f1), color='C0', alpha=0.2)
    plt.plot(z[1], np.zeros((z[1].shape)), '|C3', mew=mew)
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 5)
    plt.plot(x_plot, mean_f2, 'C0')#, plt.title('component 2')
    plt.fill_between(x_plot, mean_f2 - 2*np.sqrt(var_f2),
                             mean_f2 + 2*np.sqrt(var_f2), color='C0', alpha=0.2)
    plt.plot(z[3], np.zeros((z[3].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 6)
    plt.plot(x_plot, mean_f3, 'C0')#, plt.title('component 2')
    plt.fill_between(x_plot, mean_f3 - 2*np.sqrt(var_f3),
                             mean_f3 + 2*np.sqrt(var_f3), color='C0', alpha=0.2)
    plt.plot(z[5], np.zeros((z[5].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 7)#, plt.title('source 1')
    plt.plot(x_plot, gaussfunc(mean_g1)*mean_f1)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 8)#, plt.title('source 2')
    plt.plot(x_plot, gaussfunc(mean_g2)*mean_f2)
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 9)#, plt.title('source 2')
    plt.plot(x_plot, gaussfunc(mean_g3)*mean_f3)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, (10,12))#, plt.title('data')
    plt.plot(x_plot, y, 'C0')
    plt.plot(x_plot, gaussfunc(mean_g1)*mean_f1 + gaussfunc(mean_g2)*mean_f2 + gaussfunc(mean_g3)*mean_f3, 'k')
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    plt.suptitle(title)
    
    
    
    
    
    
    
def plot_ssgp_full(m, mean_f, var_f, mean_g, var_g, x_plot, y, title='results', parts=None):
    mean_f1, mean_f2, mean_f3 = mean_f[0].reshape(-1,), mean_f[1].reshape(-1,), mean_f[2].reshape(-1,)
    mean_g1, mean_g2, mean_g3 = mean_g[0].reshape(-1,), mean_g[1].reshape(-1,), mean_g[2].reshape(-1,)
    var_f1, var_f2, var_f3 = var_f[0].reshape(-1,), var_f[1].reshape(-1,), var_f[2].reshape(-1,)
    var_g1, var_g2, var_g3 = var_g[0].reshape(-1,), var_g[1].reshape(-1,), var_g[2].reshape(-1,)
    x_plot = x_plot.reshape(-1,)
    y_plot = y.reshape(-1,)
    xlim = [x_plot[0], x_plot[-1]]
    mew=1

    z = [m.Za1.value, m.Zc1.value, m.Za2.value, m.Zc2.value, m.Za3.value, m.Zc3.value]
    ncol, nrow = 5, 3
    plt.figure(figsize=(16, 18))
    
    plt.subplot(ncol, nrow, 1), plt.title('activation 1')
    plt.plot(x_plot, logistic(mean_g1), 'C0')
    plt.fill_between(x_plot, logistic(mean_g1-2*np.sqrt(var_g1)),
                     logistic(mean_g1+2*np.sqrt(var_g1)), color='C0', alpha=0.2)
    plt.plot(z[0], np.zeros((z[0].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 2), plt.title('activation 2')
    plt.plot(x_plot, logistic(mean_g2), 'C0')
    plt.fill_between(x_plot, logistic(mean_g2-2*np.sqrt(var_g2)),
                     logistic(mean_g2+2*np.sqrt(var_g2)), color='C0', alpha=0.2)
    plt.plot(z[2], np.zeros((z[2].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 3), plt.title('activation 3')
    plt.plot(x_plot, logistic(mean_g3), 'C0')
    plt.fill_between(x_plot, logistic(mean_g3-2*np.sqrt(var_g3)),
                     logistic(mean_g3+2*np.sqrt(var_g3)), color='C0', alpha=0.2)
    plt.plot(z[4], np.zeros((z[4].shape)), '|C3', mew=mew)
    plt.xlim(xlim)


    plt.subplot(ncol, nrow, 4), plt.title('component 1')
    plt.plot(x_plot, mean_f1, 'C0')
    plt.fill_between(x_plot, mean_f1 - 2*np.sqrt(var_f1),
                             mean_f1 + 2*np.sqrt(var_f1), color='C0', alpha=0.2)
    plt.plot(z[1], np.zeros((z[1].shape)), '|C3', mew=mew)
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 5)
    plt.plot(x_plot, mean_f2, 'C0'), plt.title('component 2')
    plt.fill_between(x_plot, mean_f2 - 2*np.sqrt(var_f2),
                             mean_f2 + 2*np.sqrt(var_f2), color='C0', alpha=0.2)
    plt.plot(z[3], np.zeros((z[3].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 6)
    plt.plot(x_plot, mean_f3, 'C0'), plt.title('component 3')
    plt.fill_between(x_plot, mean_f3 - 2*np.sqrt(var_f3),
                             mean_f3 + 2*np.sqrt(var_f3), color='C0', alpha=0.2)
    plt.plot(z[5], np.zeros((z[5].shape)), '|C3', mew=mew)
    plt.xlim(xlim)
    
    
    plt.subplot(ncol, nrow, 7), plt.title('approximate source 1')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1)
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 8), plt.title('approximate source 2')
    plt.plot(x_plot, logistic(mean_g2)*mean_f2)
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 9), plt.title('approximate source 3')
    plt.plot(x_plot, logistic(mean_g3)*mean_f3)
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 10), plt.title('real source 1')
    plt.plot(x_plot, parts[0])
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, 11), plt.title('real source 2')
    plt.plot(x_plot, parts[1])
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    
    plt.subplot(ncol, nrow, 12), plt.title('real source 3')
    plt.plot(x_plot, parts[2])
    plt.ylim([-1, 1])
    plt.xlim(xlim)

    plt.subplot(ncol, nrow, (13,15)), plt.title('data and complete approximation')
    plt.plot(x_plot, y, 'C0')
    plt.plot(x_plot, logistic(mean_g1)*mean_f1 + logistic(mean_g2)*mean_f2 + logistic(mean_g3)*mean_f3, 'k')
    plt.ylim([-1, 1])
    plt.xlim(xlim)
    plt.suptitle(title)


def plot_predict(x, mean, var, z, latent=False):    
    if latent:
        plt.plot(x, logistic(mean), 'C0', lw=2)
        plt.fill_between(x[:,0], logistic(mean[:,0] - 2*np.sqrt(var[:,0])), 
                                 logistic(mean[:,0] + 2*np.sqrt(var[:,0])), color='C0', alpha=0.2)
        
        plt.twinx()
        
        plt.plot(x, mean, 'C2', lw=2, alpha=0.5)
        plt.fill_between(x[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), 
                                 mean[:,0] + 2*np.sqrt(var[:,0]), color='C2', alpha=0.1)
    else:
        plt.plot(x, mean, 'C0', lw=2)
        plt.fill_between(x[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), 
                                 mean[:,0] + 2*np.sqrt(var[:,0]), color='C0', alpha=0.2)

    plt.plot(z, 0.*z, '|C3', mew=2)




#
