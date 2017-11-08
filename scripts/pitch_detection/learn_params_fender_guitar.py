import sys
import numpy as np
import gpflow
import soundfile
import matplotlib.pyplot as plt
sys.path.append('../../')
import gpitch
reload(gpitch)
from gpitch import myplots as mplt
plt.interactive(True)


active_device = sys.argv[1]
gpitch.amtgp.init_settings(visible_device=active_device, interactive=False)  # conf gpu usage, plot

data_loc = '../../../datasets/fender/train/'  # location of training data, where to save results
re_fil_loc = '../../../results/files/fender/'
re_fig_loc = '../../../results/figures/fender/'
lstrip, rstrip = 'm_', '.wav'
filen = gpitch.amtgp.load_filenames(data_loc, lstrip=lstrip, rstrip=rstrip)  # name of files to analyze

Np = filen.size  # number of files to process
N = 3200  # number of frames per file
Nh = 15  # number of maximum harmonics in component processes
dec, ws = 80, N  # decimation factor, and analysis window size
maxiter, restarts = 500, 3  # optimize

for i in range(0, 3):
    y, fs = soundfile.read(data_loc + lstrip + filen[i] + rstrip, frames=N, start=0)  # load data
    y = y.reshape(-1, 1) / np.max(np.abs(y))  # normalize
    x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)  # time vector
    ideal_f0 = gpitch.amtgp.midi2frec(float(filen[i]))
    F_star, S_star, F, Y, S = gpitch.amtgp.init_com_params(y=y, fs=fs, Nh=Nh, ideal_f0=ideal_f0, scaled=True)
    kc = gpitch.kernels.MaternSpecMix(lengthscales=1., variances=S_star, frequencies=F_star, Nc=Nh)
    ka = gpflow.kernels.Matern32(1)
    model = gpitch.modpdet.ModPDet(x=x, y=y, ker_com=kc, ker_act=ka, ws=ws, dec=dec)  # instantiate sigmoid model

    model.optimize(maxiter=maxiter, restarts=restarts)
    mf, vf, mg, vg, xp = model.predict_all(x)
    mplt.plot_results(mf, vf, mg, vg, xp, y)
    plt.savefig(re_fig_loc + filen[i] + '.png')
    np.savez_compressed(re_fil_loc + filen[i] + '.npz', x=x, y=y)  # save learned parameters
