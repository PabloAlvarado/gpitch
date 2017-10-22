import sys
sys.path.append('../../')
import numpy as np
from matplotlib import pyplot as plt
from gpitch.amtgp import logistic
plt.interactive(True)

def plot_results(results):
    '''
    Plot infered components and activations
    '''
    ncols = 2
    nrows = 5
    plt.figure()
    #plt.figure(figsize=(18, nrows*6))
    plt.subplot(nrows, ncols, (1, 2))
    plt.title('data and prediction')
    plt.plot(results['x_pred'], results['y_pred'], '.k', mew=1)
    plt.plot(results['x_pred'], results['yhat'] , lw=2)

    plt.subplot(nrows, ncols, (3,4))
    plt.title('source 1, pitch ' +  pitch_detect[i])
    plt.plot(results['x_pred'], logistic(results['qm2'])*results['qm1'], lw=2)

    plt.subplot(nrows, ncols, (5,6))
    plt.title('source 2')
    plt.plot(results['x_pred'], logistic(results['qm4'])*results['qm3'], lw=2)

    plt.subplot(nrows, ncols, 7)
    plt.title('activation 1')
    plt.plot(results['x_pred'], logistic(results['qm2']), 'g', lw=2)
    plt.fill_between(results['x_pred'], logistic(results['qm2']-2*np.sqrt(results['qv2'])),
                     logistic(results['qm2']+2*np.sqrt(results['qv2'])), color='g', alpha=0.2)

    plt.subplot(nrows, ncols, 8)
    plt.title('activation 2')
    plt.plot(results['x_pred'], logistic(results['qm4']), 'g', lw=2)
    plt.fill_between(results['x_pred'], logistic(results['qm4'] - 2*np.sqrt(results['qv4'])),
                     logistic(results['qm4']+2*np.sqrt(results['qv4'])), color='g', alpha=0.2)

    plt.subplot(nrows, ncols, 9)
    plt.title('component 1')
    plt.plot(results['x_pred'], results['qm1'], color='C0', lw=2)
    plt.fill_between(results['x_pred'], results['qm1']-2*np.sqrt(results['qv1']), results['qm1']+2*np.sqrt(results['qv1']),
                     color='C0', alpha=0.2)

    plt.subplot(nrows, ncols, 10)
    plt.title('component 2')
    plt.plot(results['x_pred'], results['qm3'], color='C0', lw=2)
    plt.fill_between(results['x_pred'], results['qm3']-2*np.sqrt(results['qv3']), results['qm3']+2*np.sqrt(results['qv3']),
                     color='C0', alpha=0.2)



pitch_detect = ['60', '64', '67', '72', '76']
for i in range(len(pitch_detect)):
    results = np.load('../../../results/files/demos/loogp/results_maps_pitch_'+ pitch_detect[i] + '.npz')
    plot_results(results)
    #plt.tight_layout()
    plt.savefig('../../../results/figures/demos/demo_loogp_maps_' + pitch_detect[i] + '.jpeg')





























#
