import numpy as np
import gpflow
from gpflow import settings
from gpflow.kullback_leiblers import gauss_kl, gauss_kl_white
from gpflow.minibatch import MinibatchData
import tensorflow as tf
from likelihoods import SsLik
import time
from gpitch.kernels import Matern32sm
from gpitch import get_act_params, get_com_params, get_env
from gpflow.kernels import Matern32


jitter = settings.numerics.jitter_level
float_type = settings.dtypes.float_type


def init_kernels(m, alpha=0.333):
    """Initialize kernels for source separation model"""
    var_act, ls_act = get_act_params(m.kern_act.get_parameter_dict())
    var_com, fre_com, ls_com = get_com_params(m.kern_com.get_parameter_dict())

    k_a = Matern32(input_dim=1, lengthscales=ls_act[0], variance=var_act[0])
    k_c = Matern32sm(input_dim=1, numc=fre_com.size, lengthscales=ls_com[0], variances=alpha*var_com, frequencies=fre_com)
    return k_a, k_c


def init_model(x, y, m1, m2, m3, niv_a, niv_c, minibatch_size, nlinfun):
    """Initialize pitch detection model"""
    ka1, kc1 = init_kernels(m1) 
    ka2, kc2 = init_kernels(m2)  
    ka3, kc3 = init_kernels(m3)

    nsecs = y.size/16000  # niv number inducing variables per second, duration of signal in seconds
    dec_a1 = 16000/niv_a
    dec_c1 = 16000/niv_c
    dec_a2 = 16000/niv_a
    dec_c2 = 16000/niv_c
    dec_a3 = 16000/niv_a
    dec_c3 = 16000/niv_c
    za1 = np.vstack([x[::dec_a1].copy(), x[-1].copy()])  # location inducing variables
    zc1 = np.vstack([x[::dec_c1].copy(), x[-1].copy()])  # location inducing variables
    za2 = 0.33*(za1[1] - za1[0]) + np.vstack([x[::dec_a2].copy(), x[-1].copy()])  # location inducing variables
    zc2 = 0.33*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c2].copy(), x[-1].copy()])  # location inducing variables
    za3 = 0.66*(za1[1] - za1[0]) + np.vstack([x[::dec_a3].copy(), x[-1].copy()])  # location inducing variables
    zc3 = 0.66*(zc1[1] - zc1[0]) + np.vstack([x[::dec_c3].copy(), x[-1].copy()])  # location inducing variables

    Z = [za1, zc1, za2, zc2, za3, zc3]
    m = SsGP(X=x.copy(), Y=y.copy(), kf=[kc1, kc2, kc3], kg=[ka1, ka2, ka3], Z=Z, 
             minibatch_size=minibatch_size, nlinfun=nlinfun)

    m.kern_g1.lengthscales = 0.2
    m.kern_g2.lengthscales = 0.2
    m.kern_g3.lengthscales = 0.2

    m.kern_f1.fixed = True
    m.kern_f1.lengthscales.fixed = False
    m.kern_f1.lengthscales = 1.
    
    m.kern_f2.fixed = True
    m.kern_f2.lengthscales.fixed = False
    m.kern_f2.lengthscales = 1.
    
    m.kern_f3.fixed = True
    m.kern_f3.lengthscales.fixed = False
    m.kern_f3.lengthscales = 1.

    
    m.kern_g1.variance = 1.
    m.kern_g2.variance = 1.
    m.kern_g3.variance = 1.
    
    m.kern_g1.variance.fixed = True
    m.kern_g2.variance.fixed = True
    m.kern_g3.variance.fixed = True
    
    m.likelihood.variance = 1.
    # envelope, latent, compon = get_env(y.copy(), win_size=500)
    # m.q_mu2 = np.vstack([ latent[::dec_a1].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g1
    # m.q_mu4 = np.vstack([ latent[::dec_a2].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g2
    # m.q_mu6 = np.vstack([ latent[::dec_a3].reshape(-1,1).copy(), latent[-1].reshape(-1,1).copy() ])  # g3
    return m


def predict_windowed(x, y, predfunc):
    st = time.time()
    n = y.size
    nw = 1600
    mf = [[], [], []]
    mg = [[], [], []]
    vf = [[], [], []]
    vg = [[], [], []]
    x_plot = []
    y_plot = []

    for i in range(n/nw):
        xnew = x[i*nw : (i+1)*nw].copy()
        ynew = y[i*nw : (i+1)*nw].copy()

        mean_f, mean_g, var_f, var_g = predfunc(xnew)

        mf[0].append(mean_f[0]) 
        mf[1].append(mean_f[1]) 
        mf[2].append(mean_f[2]) 

        vf[0].append(var_f[0]) 
        vf[1].append(var_f[1]) 
        vf[2].append(var_f[2])

        mg[0].append(mean_g[0]) 
        mg[1].append(mean_g[1]) 
        mg[2].append(mean_g[2])

        vg[0].append(var_g[0]) 
        vg[1].append(var_g[1]) 
        vg[2].append(var_g[2])

        x_plot.append(xnew)
        y_plot.append(ynew)
        
    mf[0] = np.asarray(mf[0]).reshape(-1, 1) 
    mf[1] = np.asarray(mf[1]).reshape(-1, 1) 
    mf[2] = np.asarray(mf[2]).reshape(-1, 1)
    vf[0] = np.asarray(vf[0]).reshape(-1, 1) 
    vf[1] = np.asarray(vf[1]).reshape(-1, 1) 
    vf[2] = np.asarray(vf[2]).reshape(-1, 1)

    mg[0] = np.asarray(mg[0]).reshape(-1, 1) 
    mg[1] = np.asarray(mg[1]).reshape(-1, 1) 
    mg[2] = np.asarray(mg[2]).reshape(-1, 1)
    vg[0] = np.asarray(vg[0]).reshape(-1, 1) 
    vg[1] = np.asarray(vg[1]).reshape(-1, 1) 
    vg[2] = np.asarray(vg[2]).reshape(-1, 1)

    x_plot = np.asarray(x_plot).reshape(-1, 1)
    y_plot = np.asarray(y_plot).reshape(-1, 1)
    
    print("Time predicting {} secs".format(time.time() - st))
    
    return mf, vf, mg, vg, x_plot, y_plot



class SsGP(gpflow.model.Model):
    def __init__(self, X, Y, kf, kg, Z, whiten=True, minibatch_size=None, nlinfun=None):
        '''Leave One Out (LOO) model.
        INPUTS:
        kf : list of kernels for each latent quasi-periodic function
        kg : list of kernels for each latent envelope function
        '''
        gpflow.model.Model.__init__(self)

        if minibatch_size is None:
            minibatch_size = X.shape[0]

        self.minibatch_size = minibatch_size
        self.num_data = X.shape[0]
        self.nlinfun = nlinfun

        self.X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        self.Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))
        
        self.Za1 = gpflow.param.Param(Z[0])
        self.Zc1 = gpflow.param.Param(Z[1])
        
        self.Za2 = gpflow.param.Param(Z[2])
        self.Zc2 = gpflow.param.Param(Z[3])
        
        self.Za3 = gpflow.param.Param(Z[4])
        self.Zc3 = gpflow.param.Param(Z[5])
        
        self.Za1.fixed = True
        self.Zc1.fixed = True
        
        self.Za2.fixed = True
        self.Zc2.fixed = True
        
        self.Za3.fixed = True
        self.Zc3.fixed = True
    
        self.num_inducing_a1 = Z[0].shape[0]
        self.num_inducing_c1 = Z[1].shape[0]
        
        self.num_inducing_a2 = Z[2].shape[0]
        self.num_inducing_c2 = Z[3].shape[0]
        
        self.num_inducing_a3 = Z[4].shape[0]
        self.num_inducing_c3 = Z[5].shape[0]


        self.kern_f1, self.kern_f2, self.kern_f3 = kf[0], kf[1], kf[2]
        self.kern_g1, self.kern_g2, self.kern_g3 = kg[0], kg[1], kg[2]
        
        
        self.likelihood = SsLik(nlinfun)
        self.whiten = whiten

        # initialize variational parameters
        self.q_mu1 = gpflow.param.Param(np.zeros((self.Zc1.shape[0], 1)))  # f1
        self.q_mu2 = gpflow.param.Param(-np.ones((self.Za1.shape[0], 1)))  # g1
        self.q_mu3 = gpflow.param.Param(np.zeros((self.Zc2.shape[0], 1)))  # f2
        self.q_mu4 = gpflow.param.Param(-np.ones((self.Za2.shape[0], 1)))  # g2
        self.q_mu5 = gpflow.param.Param(np.zeros((self.Zc2.shape[0], 1)))  # f3
        self.q_mu6 = gpflow.param.Param(-np.ones((self.Za2.shape[0], 1)))  # g3

        q_sqrt_a1 = np.array([np.eye(self.num_inducing_a1) for _ in range(1)]).swapaxes(0, 2)
        q_sqrt_c1 = np.array([np.eye(self.num_inducing_c1) for _ in range(1)]).swapaxes(0, 2)
        q_sqrt_a2 = np.array([np.eye(self.num_inducing_a2) for _ in range(1)]).swapaxes(0, 2)
        q_sqrt_c2 = np.array([np.eye(self.num_inducing_c2) for _ in range(1)]).swapaxes(0, 2)
        q_sqrt_a3 = np.array([np.eye(self.num_inducing_a3) for _ in range(1)]).swapaxes(0, 2)
        q_sqrt_c3 = np.array([np.eye(self.num_inducing_c3) for _ in range(1)]).swapaxes(0, 2)

        self.q_sqrt1 = gpflow.param.Param(q_sqrt_c1.copy())
        self.q_sqrt2 = gpflow.param.Param(q_sqrt_a1.copy())
        self.q_sqrt3 = gpflow.param.Param(q_sqrt_c2.copy())
        self.q_sqrt4 = gpflow.param.Param(q_sqrt_a2.copy())
        self.q_sqrt5 = gpflow.param.Param(q_sqrt_c3.copy())
        self.q_sqrt6 = gpflow.param.Param(q_sqrt_a3.copy())


    def build_prior_KL(self):
        if self.whiten:
            KL1 = gauss_kl_white(self.q_mu1, self.q_sqrt1)
            KL2 = gauss_kl_white(self.q_mu2, self.q_sqrt2)
            KL3 = gauss_kl_white(self.q_mu3, self.q_sqrt3)
            KL4 = gauss_kl_white(self.q_mu4, self.q_sqrt4)
            KL5 = gauss_kl_white(self.q_mu5, self.q_sqrt5)
            KL6 = gauss_kl_white(self.q_mu6, self.q_sqrt6)
        else:
            K1 = self.kern_f1.K(self.Zc1) + tf.eye(self.num_inducing_c1, dtype=float_type) * jitter
            K2 = self.kern_g1.K(self.Za1) + tf.eye(self.num_inducing_a1, dtype=float_type) * jitter
            K3 = self.kern_f2.K(self.Zc2) + tf.eye(self.num_inducing_c2, dtype=float_type) * jitter
            K4 = self.kern_g2.K(self.Za2) + tf.eye(self.num_inducing_a2, dtype=float_type) * jitter
            K5 = self.kern_g2.K(self.Zc3) + tf.eye(self.num_inducing_c3, dtype=float_type) * jitter
            K6 = self.kern_g2.K(self.Za3) + tf.eye(self.num_inducing_a3, dtype=float_type) * jitter
            KL1 = gauss_kl(self.q_mu1, self.q_sqrt1, K1)
            KL2 = gauss_kl(self.q_mu2, self.q_sqrt2, K2)
            KL3 = gauss_kl(self.q_mu3, self.q_sqrt3, K3)
            KL4 = gauss_kl(self.q_mu4, self.q_sqrt4, K4)
            KL5 = gauss_kl(self.q_mu5, self.q_sqrt5, K5)
            KL6 = gauss_kl(self.q_mu6, self.q_sqrt6, K6)
        #aux0 = tf.reduce_max(tf.abs(self.q_mu1))
        return KL1 + KL2 + KL3 + KL4 + KL5 + KL6 #+ 100.*tf.abs(aux0 - 1.)

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean1, fvar1 = gpflow.conditionals.conditional(self.X, self.Zc1,
                                                        self.kern_f1, self.q_mu1,
                                                        q_sqrt=self.q_sqrt1,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        
        fmean2, fvar2 = gpflow.conditionals.conditional(self.X, self.Za1,
                                                        self.kern_g1, self.q_mu2,
                                                        q_sqrt=self.q_sqrt2,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        
        fmean3, fvar3 = gpflow.conditionals.conditional(self.X, self.Zc2,
                                                        self.kern_f2, self.q_mu3,
                                                        q_sqrt=self.q_sqrt3,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        
        fmean4, fvar4 = gpflow.conditionals.conditional(self.X, self.Za2,
                                                        self.kern_g2, self.q_mu4,
                                                        q_sqrt=self.q_sqrt4,
                                                        full_cov=False,
                                                        whiten=self.whiten)
                
        fmean5, fvar5 = gpflow.conditionals.conditional(self.X, self.Zc3,
                                                        self.kern_f3, self.q_mu5,
                                                        q_sqrt=self.q_sqrt5,
                                                        full_cov=False,
                                                        whiten=self.whiten)
        
        fmean6, fvar6 = gpflow.conditionals.conditional(self.X, self.Za3,
                                                        self.kern_g3, self.q_mu6,
                                                        q_sqrt=self.q_sqrt6,
                                                        full_cov=False,
                                                        whiten=self.whiten)   
        
        fmean = tf.concat([fmean1, fmean2, fmean3, fmean4, fmean5, fmean6], 1)
        fvar = tf.concat([fvar1, fvar2, fvar3, fvar4, fvar5, fvar6], 1)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL
    
    
    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predictall(self, Xnew):
        
        mf1, vf1 =  gpflow.conditionals.conditional(Xnew, self.Zc1, self.kern_f1,
                                               self.q_mu1, q_sqrt=self.q_sqrt1,
                                               full_cov=False, whiten=self.whiten)

        mg1, vg1 =  gpflow.conditionals.conditional(Xnew, self.Za1, self.kern_g1,
                                               self.q_mu2, q_sqrt=self.q_sqrt2,
                                               full_cov=False, whiten=self.whiten)

        mf2, vf2 = gpflow.conditionals.conditional(Xnew, self.Zc2, self.kern_f2,
                                               self.q_mu3, q_sqrt=self.q_sqrt3,
                                               full_cov=False, whiten=self.whiten)

        mg2, vg2 = gpflow.conditionals.conditional(Xnew, self.Za2, self.kern_g2,
                                               self.q_mu4, q_sqrt=self.q_sqrt4,
                                               full_cov=False, whiten=self.whiten)

        mf3, vf3 = gpflow.conditionals.conditional(Xnew, self.Zc3, self.kern_f3,
                                               self.q_mu5, q_sqrt=self.q_sqrt5,
                                               full_cov=False, whiten=self.whiten)

        mg3, vg3 =  gpflow.conditionals.conditional(Xnew, self.Za3, self.kern_g3,
                                               self.q_mu6, q_sqrt=self.q_sqrt6,
                                               full_cov=False, whiten=self.whiten)
        mf = [mf1, mf2, mf3]
        mg = [mg1, mg2, mg3]
        vf = [vf1, vf2, vf3]
        vg = [vg1, vg2, vg3]
        return mf, mg, vf, vg

    
#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_f1(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Zc1, self.kern_f1,
#                                                self.q_mu1, q_sqrt=self.q_sqrt1,
#                                                full_cov=False, whiten=self.whiten)

#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_g1(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Za1, self.kern_g1,
#                                                self.q_mu2, q_sqrt=self.q_sqrt2,
#                                                full_cov=False, whiten=self.whiten)

#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_f2(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Zc2, self.kern_f2,
#                                                self.q_mu3, q_sqrt=self.q_sqrt3,
#                                                full_cov=False, whiten=self.whiten)

#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_g2(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Za2, self.kern_g2,
#                                                self.q_mu4, q_sqrt=self.q_sqrt4,
#                                                full_cov=False, whiten=self.whiten)

#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_f3(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Zc3, self.kern_f3,
#                                                self.q_mu5, q_sqrt=self.q_sqrt5,
#                                                full_cov=False, whiten=self.whiten)

#     @gpflow.param.AutoFlow((tf.float64, [None, None]))
#     def predict_g3(self, Xnew):
#         return gpflow.conditionals.conditional(Xnew, self.Za3, self.kern_g3,
#                                                self.q_mu6, q_sqrt=self.q_sqrt6,
#                                                full_cov=False, whiten=self.whiten)
