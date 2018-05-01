import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

def plot_prediction(x, y, source, m_a, v_a, m_c, v_c, m, nlinfun):
    numf = len(source)
    for i in range(numf):
        plt.figure(figsize=(16, 4))
        plt.subplot(1,4,1)
        plt.plot(x[i], y[i])
        plt.ylim([-1.2, 1.2])

        plt.subplot(1,4,2)
        plt.plot(x[i], source[i])
        plt.ylim([-1.2, 1.2])

        plt.subplot(1,4,3)
        plt.plot(x[i], nlinfun(m_a[i]), m[i].za[0].value, 0.*m[i].za[0].value, 'k|')
        plt.plot(x[i], nlinfun(m_a[i] + 2*np.sqrt(v_a[i])), 'C0--', alpha=0.7)
        plt.plot(x[i], nlinfun(m_a[i] - 2*np.sqrt(v_a[i])), 'C0--', alpha=0.7)
        plt.twinx()
        plt.plot(x[i], m_a[i], 'g', alpha=0.5)
        plt.plot(x[i], m_a[i] + 2*np.sqrt(v_a[i]), 'g--', alpha=0.3)
        plt.plot(x[i], m_a[i] - 2*np.sqrt(v_a[i]), 'g--', alpha=0.3)

        plt.subplot(1,4,4)
        plt.plot(x[i], m_c[i])
        plt.plot(x[i], m_c[i] - 2*np.sqrt(v_c[i]), 'C0--', alpha=0.7)
        plt.plot(x[i], m_c[i] + 2*np.sqrt(v_c[i]), 'C0--', alpha=0.7)
        plt.plot(m[i].zc[0].value, 0.*m[i].zc[0].value, 'k|')
        
        
def plot_fft(F, y, y_k, numf, iparam):
    plt.figure(figsize=(16, 16))  # visualize loaded data 
    for i in range(numf):
        Y1 = np.abs(fft(y[i].reshape(-1, ) ) )[0:16000]
        Y2 = np.abs(fft(y_k[i].reshape(-1, ) ) )[0:16000]
        Y1 /= np.max(Y1)
        Y2 /= np.max(Y2)

        plt.subplot(4, 3, i+1)
        plt.plot(F, Y1 , 'C0' )
        plt.plot(F+40., Y2, 'C1' )
        plt.twinx()
        plt.plot(20. + iparam[i][0], iparam[i][1] / np.max(iparam[i][1]), '|C4', mew=2)
        plt.xlim(0, 4000)

        
def plot_parameters(m):
    plt.figure(figsize=(16, 4))
    nr, nc = 1, 5
    plt.subplot(nr, nc, 1), plt.title('lengthscale activation'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_act[0].lengthscales.value, '.C1')
    plt.xlim(-1, 12), plt.ylim([0, 2.5])

    plt.subplot(nr, nc, 2), plt.title('variance activation'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_act[0].variance.value, '.C1')
    plt.xlim(-1, 12), plt.ylim([0, 5.])

    plt.subplot(nr, nc, 3), plt.title('lengthscale component'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_com[0].lengthscales.value, 'C1.')
    plt.xlim(-1, 12), plt.ylim([0, 5.])

    plt.subplot(nr, nc, 4), plt.title('f0 component'), plt.grid(True) 
    for i in range(len(m)):
        plt.plot(i, m[i].kern_com[0].frequency_1.value, 'C1.')
    plt.xlim(-1, 12), plt.ylim([200, 500])

    plt.subplot(nr, nc, 5), plt.title('noise variance'), plt.grid(True) 
    for i in range(len(m)):
        plt.plot(i, m[i].likelihood.variance.value, 'C1.')
    plt.xlim(-1, 12), plt.ylim([-0.001, 0.005])