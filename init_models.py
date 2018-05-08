from gpflow.kernels import Matern32, Matern12
from gpitch.kernels import Matern32sm
from gpitch.methods import find_ideal_f0, init_cparam
import numpy as np


def init_iv(x, num_sources, nivps_a, nivps_c, fs=16000):
    """
    Initialize inducing variables
    :param x: time vector
    :param fs: sample frequency
    :param nivps_a: number inducing variables per second for activation
    :param nivps_c: number inducing variables per second for component
    """
    dec_a = fs/nivps_a
    dec_c = fs/nivps_c
    za = []
    zc = []
    for i in range(num_sources):
        za.append(np.vstack([x[::dec_a].copy(), x[-1].copy()]))  # location ind v act
        zc.append(np.vstack([x[::dec_c].copy(), x[-1].copy()]))  # location ind v comp
    z = [za, zc]
    return z

def init_kernel_training(y, list_files, fs=16000, maxh=25):
    num_pitches = len(list_files)
    if0 = find_ideal_f0(list_files)  # ideal frequency for each pitch
    iparam = []  # initilize component kernel parameters for each pitch model
    kern_act = []
    kern_com = []
    for i in range(num_pitches):
        iparam.append(init_cparam(y[i], fs=fs, maxh=maxh, ideal_f0=if0[i])) # init component kern params

        kern_act.append(Matern12(1, lengthscales=1., variance=3.5))
        kern_com.append(Matern32sm(1, num_partials=len(iparam[i][1]), lengthscales=1., variances=iparam[i][1],
                                   frequencies=iparam[i][0]))
        kern_com[i].vars_n_freqs_fixed()

    kern = [kern_act, kern_com]
    return kern, iparam # list of all required kernels and its initial parameters

def init_kernel_with_trained_models(m, option_two=False):
    kern_act = []
    kern_com = []
    num_sources = len(m)
    for i in range(num_sources):
        num_p = m[i].kern_com[0].num_partials
        kern_act.append(Matern12(1))
        kern_com.append(Matern32sm(1, num_partials=num_p))
                
        kern_act[i].fixed = True
        kern_com[i].fixed = True
        kern_com[i].vars_n_freqs_fixed(fix_var=False, fix_freq=False)
        
        if option_two:
            kern_act[i].lengthscales = 0.5
            kern_act[i].variance = 4.0
            kern_com[i].lengthscales = 1.0
        else:
            kern_act[i].lengthscales = m[i].kern_act[0].lengthscales.value.copy()
            kern_act[i].variance = m[i].kern_act[0].variance.value.copy()
            kern_com[i].lengthscales = m[i].kern_com[0].lengthscales.value.copy()
        
        kern_act[i].fixed = False
        kern_com[i].lengthscales.fixed = False
        
        for j in range(num_p):
            kern_com[i].frequency[j] = m[i].kern_com[0].frequency[j].value.copy()
            kern_com[i].variance[j] = m[i].kern_com[0].variance[j].value.copy()
    return [kern_act, kern_com]

def reset_model(m, x, y, nivps, m_trained, option_two=False):
    num_sources = len(m.za)
    m.x = x.copy()
    m.y = y.copy()
    m.likelihood.variance = 1.    
    new_z = init_iv(x, num_sources, nivps[0], nivps[1])
    for i in range(num_sources):
        m.za[i] = new_z[0][i].copy()
        m.zc[i] = new_z[1][i].copy()
        
        m.q_mu_act[i] = np.zeros((new_z[0][i].shape[0], 1))
        m.q_mu_com[i] = np.zeros((new_z[1][i].shape[0], 1))
        
        m.q_sqrt_act[i] = np.array([np.eye(new_z[0][i].shape[0]) for _ in range(1)]).swapaxes(0, 2)
        m.q_sqrt_com[i] = np.array([np.eye(new_z[1][i].shape[0]) for _ in range(1)]).swapaxes(0, 2)
        
        
        if option_two:
            m.kern_act[i].lengthscales = 0.5
            m.kern_act[i].variance = 4.0
            m.kern_com[i].lengthscales = 1.0
        else:    
            m.kern_act[i].lengthscales = m_trained[i].kern_act[0].lengthscales.value.copy()
            m.kern_act[i].variance = m_trained[i].kern_act[0].variance.value.copy()
            m.kern_com[i].lengthscales = m_trained[i].kern_com[0].lengthscales.value.copy()
        
        num_p = m.kern_com[i].num_partials
        for j in range(num_p):
            m.kern_com[i].frequency[j] = m_trained[i].kern_com[0].frequency[j].value.copy()
            m.kern_com[i].variance[j] = m_trained[i].kern_com[0].variance[j].value.copy()
       
        
    


# def re_init_params(m, x, y, nivps):

#     # reset inducing variables
#     dec_a = 16000/nivps[0]
#     dec_c = 16000/nivps[1]
#     za1 = np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
#     zc1 = np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
#     za2 = 0.33*(za1[1] - za1[0]) + np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
#     zc2 = 0.33*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
#     za3 = 0.66*(za1[1] - za1[0]) + np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
#     zc3 = 0.66*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables

#     m.Za1 = za1.copy()
#     m.Za2 = za2.copy()
#     m.Za3 = za3.copy()
#     m.Zc1 = zc1.copy()
#     m.Zc2 = zc2.copy()
#     m.Zc3 = zc3.copy()

#     # reset input data
#     m.X = x.copy()
#     m.Y = y.copy()

#     # reset variational parameters
#     m.q_mu1 = np.zeros((zc1.shape[0], 1))  # f1
#     m.q_mu2 = np.zeros((za1.shape[0], 1))  # g1
#     m.q_mu3 = np.zeros((zc2.shape[0], 1))  # f2
#     m.q_mu4 = np.zeros((za2.shape[0], 1))  # g2
#     m.q_mu5 = np.zeros((zc3.shape[0], 1))  # f3
#     m.q_mu6 = np.zeros((za3.shape[0], 1))  # g3

#     q_sqrt_a1 = np.array([np.eye(za1.shape[0]) for _ in range(1)]).swapaxes(0, 2)
#     q_sqrt_c1 = np.array([np.eye(zc1.shape[0]) for _ in range(1)]).swapaxes(0, 2)
#     q_sqrt_a2 = np.array([np.eye(za2.shape[0]) for _ in range(1)]).swapaxes(0, 2)
#     q_sqrt_c2 = np.array([np.eye(zc2.shape[0]) for _ in range(1)]).swapaxes(0, 2)
#     q_sqrt_a3 = np.array([np.eye(za3.shape[0]) for _ in range(1)]).swapaxes(0, 2)
#     q_sqrt_c3 = np.array([np.eye(zc3.shape[0]) for _ in range(1)]).swapaxes(0, 2)

#     m.q_sqrt1 = q_sqrt_c1.copy()
#     m.q_sqrt2 = q_sqrt_a1.copy()
#     m.q_sqrt3 = q_sqrt_c2.copy()
#     m.q_sqrt4 = q_sqrt_a2.copy()
#     m.q_sqrt5 = q_sqrt_c3.copy()
#     m.q_sqrt6 = q_sqrt_a3.copy()

#     # reset hyper-parameters
#     m.kern_g1.variance = 4.
#     m.kern_g2.variance = 4.
#     m.kern_g3.variance = 4.

#     m.kern_g1.lengthscales = 0.5
#     m.kern_g2.lengthscales = 0.5
#     m.kern_g3.lengthscales = 0.5

#     m.kern_f1.lengthscales = 1.0
#     m.kern_f2.lengthscales = 1.0
#     m.kern_f3.lengthscales = 1.0

#     m.likelihood.variance = 1.


# def get_lists_save_results():
#     return [], [], [], [], [], [], [[], [], []], [[], [], []], [[], [], []], [[], [], []]



























# def get_act_params(indict):
#     """Get parameters of activation kernel"""
#     ls = [indict[indict.keys()[i]].copy()  for i in range(len(indict)) if indict.keys()[i].endswith('lengthscales')]
#     var = [indict[indict.keys()[i]].copy() for i in range(len(indict)) if indict.keys()[i].endswith('variance')]
#     ls = np.asarray(ls).reshape(-1,)
#     var = np.asarray(var).reshape(-1,)
#     return var, ls

# def get_com_params(indict):
#     """Get parameters of activation kernel"""
#     n = len(indict)
#     var = []
#     fre = []
#     for j in range((n - 1)/2 ):
#         var.append( [indict[indict.keys()[i]].copy() for i in range(n) if indict.keys()[i].endswith('variance_' + str(j+1))])
#         fre.append( [indict[indict.keys()[i]].copy() for i in range(n) if indict.keys()[i].endswith('frequency_' + str(j+1))])
#     ls = [indict[indict.keys()[i]].copy()  for i in range(len(indict)) if indict.keys()[i].endswith('lengthscales')]
#     var = np.asarray(var).reshape(-1,1)
#     fre = np.asarray(fre).reshape(-1,1)
#     ls = np.asarray(ls).reshape(-1,)
#     return var, fre, ls

# def init_kernels_pd(m, background=False, alpha=1.):
#     """Initialize kernels for pitch detection model"""
#     if background:
#         alpha = 0.5
#     var_act, ls_act = get_act_params(m.kern_act.get_parameter_dict())
#     var_com, fre_com, ls_com = get_com_params(m.kern_com.get_parameter_dict())

#     k_a = Matern32(input_dim=1, lengthscales=ls_act[0], variance=var_act[0])
#     k_c = Matern32sm(input_dim=1, numc=fre_com.size, lengthscales=ls_com[0], variances=alpha*var_com, frequencies=fre_com)
#     return k_a, k_c

# def init_model_pd(x, y, m1, m2, m3, niv=20, minibatch_size=475):
#     """Initialize pitch detection model"""
#     ka1, kc1 = init_kernels_pd(m1, background=False)  # kernels for pitch to detect
#     ka2, kc2 = init_kernels_pd(m2, background=True)  # kernels for background
#     ka3, kc3 = init_kernels_pd(m3, background=True)

#     k_bg = kc2 + kc3

#     nsecs = y.size/16000  # niv number inducing variables per second, duration of signal in seconds
#     dec = 16000/niv
#     z = np.vstack([x[::dec].copy(), x[-1].copy()])  # location inducing variables
#     #z = np.linspace(x[0], x[-1], niv*nsecs).reshape(-1, 1)
#     m = loogp.LooGP(X=x.copy(), Y=y.copy(), kf=[kc1, k_bg], kg=[ka1, ka2], Z=z, minibatch_size=minibatch_size)

#     envelope, latent, compon = get_env(y.copy(), win_size=500)
#     m.q_mu1 = np.vstack([ compon[::dec].reshape(-1,1).copy(), compon[-1].reshape(-1,1).copy() ])  # f1
#     m.q_mu2 = np.vstack([ latent[::dec].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g1
#     m.q_mu3 = np.vstack([ compon[::dec].reshape(-1,1).copy(), compon[-1].reshape(-1,1).copy() ])  # f2
#     m.q_mu4 = np.vstack([ latent[::dec].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g2
#     m.kern_g1.lengthscales = 0.05
#     m.kern_g2.lengthscales = 0.05

#     m.kern_f1.fixed = True
#     m.kern_f1.lengthscales.fixed = False
#     m.kern_f1.lengthscales = 1.

#     m.kern_f2.fixed = True
#     m.kern_f2.matern32specmix_1.lengthscales.fixed = False
#     m.kern_f2.matern32specmix_1.lengthscales = 1.
#     m.kern_f2.matern32specmix_2.lengthscales.fixed = False
#     m.kern_f2.matern32specmix_2.lengthscales = 1.

#     m.kern_g1.variance.fixed = True
#     m.kern_g2.variance.fixed = True

#     return m

# # def plot_loaded_models(m, instr_name):
# #     for i in range(len(m)):
# #         x = m[i].x.value.copy()
# #         y = m[i].y.value.copy()
# #         z = m[i].z.value.copy()
# #         xplot = x.reshape(-1, ).copy()
# #         mean_g, var_g = m[i].prediction_act
# #         mean_f, var_f = m[i].prediction_com
# #         myplots.plot_results(mean_f.reshape(-1,), var_f.reshape(-1,), mean_g.reshape(-1,), var_g.reshape(-1,), xplot, y, z, xlim=[-0.01, 1.01])
# #         plt.suptitle(instr_name)

# def get_env(f, win_size=160):
#     """get envelope and component of a function f"""
#     win = signal.hann(win_size)
#     filtered = signal.convolve(np.abs(f).reshape(-1,), win, mode='same') / sum(win)
#     filtered = filtered/np.max(np.abs(filtered))
#     latent = ilogistic(0.001 + 0.9*filtered)
#     comp = f*(1./(filtered.reshape(-1,1) + 0.000001))
#     return filtered, latent, comp



# def opti(y, m, dec, maxiter, init_maxiter=200):
#     """optimize intializing with approximate envelope"""
#     env, lat, comp = get_env(y)
#     m.q_mu_act = np.vstack([lat[::dec[0]].reshape(-1,1).copy(), lat[-1].reshape(-1,1).copy()])


#     m.q_mu_act.fixed = True
#     m.q_sqrt_act.fixed = True
#     m.kern_act.lengthscales.fixed = True
#     m.kern_act.variance.fixed = True

#     m.kern_com.lengthscales.fixed = False
#     m.q_mu_com.fixed = False
#     m.q_sqrt_com.fixed = False
#     m.likelihood.variance.fixed = False

#     m.optimize(disp=1, maxiter=init_maxiter)

#     m.q_mu_act.fixed = False
#     m.q_sqrt_act.fixed = False
#     m.kern_act.lengthscales.fixed = False

#     m.optimize(disp=1, maxiter=maxiter)



# def init_model_pd_loo2(x, y, m1, m2, m3, niv_a=10, niv_c=50, minibatch_size=475):
#     """Initialize pitch detection model"""
#     ka1, kc1 = init_kernels_pd(m1, background=False)  # kernels for pitch to detect
#     ka2, kc2 = init_kernels_pd(m2, background=True)  # kernels for background
#     ka3, kc3 = init_kernels_pd(m3, background=True)

#     k_bg = kc2 + kc3

#     nsecs = y.size/16000  # niv number inducing variables per second, duration of signal in seconds
#     dec_a = 16000/niv_a
#     dec_c = 16000/niv_c
#     za = np.vstack([x[::dec_a].copy(), x[-1].copy()])  # location inducing variables
#     zc = np.vstack([x[::dec_c].copy(), x[-1].copy()])  # location inducing variables
#     #z = np.linspace(x[0], x[-1], niv*nsecs).reshape(-1, 1)
#     m = loogp2.LooGP2(X=x.copy(), Y=y.copy(), kf=[kc1, k_bg], kg=[ka1, ka2], Za=za, Zc=zc, minibatch_size=minibatch_size)

#     envelope, latent, compon = get_env(y.copy(), win_size=500)
#     #m.q_mu1 = np.vstack([ compon[::dec_c].reshape(-1,1).copy(), compon[-1].reshape(-1,1).copy() ])  # f1
#     m.q_mu2 = np.vstack([ latent[::dec_a].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g1
#     #m.q_mu3 = np.vstack([ compon[::dec_c].reshape(-1,1).copy(), compon[-1].reshape(-1,1).copy() ])  # f2
#     m.q_mu4 = np.vstack([ latent[::dec_a].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g2
#     m.kern_g1.lengthscales = 0.1
#     m.kern_g2.lengthscales = 0.1

#     m.kern_f1.fixed = True
#     m.kern_f1.lengthscales.fixed = False
#     m.kern_f1.lengthscales = 1.

#     m.kern_f2.fixed = True
#     m.kern_f2.matern32sm_1.lengthscales.fixed = False
#     m.kern_f2.matern32sm_1.lengthscales = 1.
#     m.kern_f2.matern32sm_2.lengthscales.fixed = False
#     m.kern_f2.matern32sm_2.lengthscales = 1.

#     m.kern_g1.variance.fixed = True
#     m.kern_g2.variance.fixed = True

#     return m

# # def plot_loaded_models_2(m, instr_name):
# #     for i in range(len(m)):
# #         x = m[i].x.value.copy()
# #         y = m[i].y.value.copy()
# #         za = m[i].za.value.copy()
# #         zc = m[i].zc.value.copy()
# #         xplot = x.reshape(-1, ).copy()
# #         mean_g, var_g = m[i].prediction_act
# #         mean_f, var_f = m[i].prediction_com
# #         myplots.plot_results_2(mean_f.reshape(-1,), var_f.reshape(-1,), mean_g.reshape(-1,), var_g.reshape(-1,), xplot, y, za, zc, xlim=[-0.01, 1.01])
# #         plt.suptitle(instr_name)













#
