import numpy as np
import loogp
import myplots
from matplotlib import pyplot as plt
from gpflow.kernels import Matern32
from gpitch.kernels import Matern32SpecMix


def get_act_params(indict):
    """Get parameters of activation kernel"""
    ls = [indict[indict.keys()[i]].copy()  for i in range(len(indict)) if indict.keys()[i].endswith('lengthscales')]
    var = [indict[indict.keys()[i]].copy() for i in range(len(indict)) if indict.keys()[i].endswith('variance')]
    ls = np.asarray(ls).reshape(-1,)
    var = np.asarray(var).reshape(-1,)
    return var, ls

def get_com_params(indict):
    """Get parameters of activation kernel"""
    n = len(indict)
    var = []
    fre = []
    for j in range((n - 1)/2 ):
        var.append( [indict[indict.keys()[i]].copy() for i in range(n) if indict.keys()[i].endswith('variance_' + str(j+1))])
        fre.append( [indict[indict.keys()[i]].copy() for i in range(n) if indict.keys()[i].endswith('frequency_' + str(j+1))])
    ls = [indict[indict.keys()[i]].copy()  for i in range(len(indict)) if indict.keys()[i].endswith('lengthscales')]
    var = np.asarray(var).reshape(-1,1)
    fre = np.asarray(fre).reshape(-1,1)
    ls = np.asarray(ls).reshape(-1,)
    return var, fre, ls

def init_kernels_pd(m, background=False, alpha=1.):
    """Initialize kernels for pitch detection model"""
    if background:
        alpha = 0.5
    var_act, ls_act = get_act_params(m.kern_act.get_parameter_dict())
    var_com, fre_com, ls_com = get_com_params(m.kern_com.get_parameter_dict())

    k_a = Matern32(input_dim=1, lengthscales=ls_act[0], variance=var_act[0])
    k_c = Matern32SpecMix(input_dim=1, numc=fre_com.size, lengthscales=ls_com[0], variances=alpha*var_com, frequencies=fre_com)
    return k_a, k_c

def init_model_pd(x, y, m1, m2, m3):
    """Initialize pitch detection model"""
    ka1, kc1 = init_kernels_pd(m1, background=False)  # kernels for pitch to detect
    ka2, kc2 = init_kernels_pd(m2, background=True)  # kernels for background
    ka3, kc3 = init_kernels_pd(m3, background=True)

    k_bg = kc2 + kc3

    niv, nsecs = 20, y.size/16000  # number inducong variables, duration of signal in seconds
    z = np.linspace(x[0], x[-1], niv*nsecs).reshape(-1, 1)
    m = loogp.LooGP(X=x.copy(), Y=y.copy(), kf=[kc1, k_bg], kg=[ka1, ka2], Z=z, minibatch_size=475)

    m.kern_f1.fixed = True
    m.kern_f1.lengthscales.fixed = False
    m.kern_f1.lengthscales = 1.

    m.kern_f2.fixed = True
    m.kern_f2.matern32specmix_1.lengthscales.fixed = False
    m.kern_f2.matern32specmix_1.lengthscales = 1.
    m.kern_f2.matern32specmix_2.lengthscales.fixed = False
    m.kern_f2.matern32specmix_2.lengthscales = 1.

    m.kern_g1.variance.fixed = True
    m.kern_g2.variance.fixed = True

    return m

def plot_loaded_models(m, instr_name):
    for i in range(len(m)):
        x = m[i].x.value.copy()
        y = m[i].y.value.copy()
        z = m[i].z.value.copy()
        xplot = x.reshape(-1, ).copy()
        mean_g, var_g = m[i].prediction_act
        mean_f, var_f = m[i].prediction_com
        myplots.plot_results(mean_f.reshape(-1,), var_f.reshape(-1,), mean_g.reshape(-1,), var_g.reshape(-1,), xplot, y, z, xlim=[-0.01, 1.01])
        plt.suptitle(instr_name)






















#
