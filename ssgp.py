import numpy as np
import gpflow
from gpflow import settings
from gpflow.kullback_leiblers import gauss_kl, gauss_kl_white
from gpflow.minibatch import MinibatchData
import tensorflow as tf
from likelihoods import SsLik
import time

jitter = settings.numerics.jitter_level
float_type = settings.dtypes.float_type

class SsGP(gpflow.model.Model):
    def __init__(self, X, Y, kf, kg, Z, whiten=True, minibatch_size=None,
                 old_version=False):
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

        self.X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        self.Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))
        #self.X = gpflow.param.DataHolder(X, on_shape_change='pass')
        #self.Y = gpflow.param.DataHolder(Y, on_shape_change='pass')

        self.Za1 = gpflow.param.DataHolder(Z[0], on_shape_change='pass')
        self.Zc1 = gpflow.param.DataHolder(Z[1], on_shape_change='pass')
        
        self.Za2 = gpflow.param.DataHolder(Z[2], on_shape_change='pass')
        self.Zc2 = gpflow.param.DataHolder(Z[3], on_shape_change='pass')
        
        self.Za3 = gpflow.param.DataHolder(Z[4], on_shape_change='pass')
        self.Zc3 = gpflow.param.DataHolder(Z[5], on_shape_change='pass')
        
        self.num_inducing_a1 = self.Za1.shape[0]
        self.num_inducing_c1 = self.Zc1.shape[0]
        
        self.num_inducing_a2 = self.Za2.shape[0]
        self.num_inducing_c2 = self.Zc2.shape[0]
        
        self.num_inducing_a3 = self.Za3.shape[0]
        self.num_inducing_c3 = self.Zc3.shape[0]



        self.kern_f1, self.kern_f2, self.kern_f3 = kf[0], kf[1], kf[2]
        self.kern_g1, self.kern_g2, self.kern_g3 = kg[0], kg[1], kg[2]

        self.likelihood = SsLik(version=old_version)
        self.whiten = whiten

        # initialize variational parameters
        self.q_mu1 = gpflow.param.Param(np.zeros((self.Zc1.shape[0], 1)))  # f1
        self.q_mu2 = gpflow.param.Param(np.zeros((self.Za1.shape[0], 1)))  # g1
        self.q_mu3 = gpflow.param.Param(np.zeros((self.Zc2.shape[0], 1)))  # f2
        self.q_mu4 = gpflow.param.Param(np.zeros((self.Za2.shape[0], 1)))  # g2
        self.q_mu5 = gpflow.param.Param(np.zeros((self.Zc2.shape[0], 1)))  # f3
        self.q_mu6 = gpflow.param.Param(np.zeros((self.Za2.shape[0], 1)))  # g3

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

    
#     def predict_all(self, xnew):
#         """
#         method introduced by Pablo A. Alvarado (14/11/2017)

#         This method call all the decorators needed to compute the prediction over the latent
#         components and activations. It also reshape the arrays to make easier to plot the
#         intervals of confidency.
#         """
#         n = xnew.size # total number of samples
#         spw = 1600 # number of samples per window
#         nw =  n/spw  # total number of windows
#         l_xnew = [ xnew[spw*i : spw*(i+1)].copy() for i in range(nw) ]
#         l_com_1_mean = []  # list to storage predictions
#         l_com_1_var = []
#         l_com_2_mean = []
#         l_com_2_var = []
#         l_com_3_mean = []
#         l_com_3_var = []

#         l_act_1_mean = []
#         l_act_1_var = []
#         l_act_2_mean = []
#         l_act_2_var = []
#         l_act_3_mean = []
#         l_act_3_var = []

#         for i in range(len(l_xnew)):
#             mean_f1, var_f1 = self.predict_f1(l_xnew[i])
#             mean_g1, var_g1 = self.predict_g1(l_xnew[i])
            
#             mean_f2, var_f2 = self.predict_f2(l_xnew[i])
#             mean_g2, var_g2 = self.predict_g2(l_xnew[i])
            
#             mean_f3, var_f3 = self.predict_f3(l_xnew[i])
#             mean_g3, var_g3 = self.predict_g3(l_xnew[i])
            
#             l_com_1_mean.append(mean_f1)
#             l_com_1_var.append(var_f1)
#             l_com_2_mean.append(mean_f2)
#             l_com_2_var.append(var_f2)
#             l_com_3_mean.append(mean_f3)
#             l_com_3_var.append(var_f3)

#             l_act_1_mean.append(mean_g1)
#             l_act_1_var.append(var_g1)
#             l_act_2_mean.append(mean_g2)
#             l_act_2_var.append(var_g2)
#             l_act_3_mean.append(mean_g3)
#             l_act_3_var.append(var_g3)
            
#         mean_f1 = np.asarray(l_com_1_mean).reshape(-1,)
#         var_f1 = np.asarray(l_com_1_var).reshape(-1,)
#         mean_f2 = np.asarray(l_com_2_mean).reshape(-1,)
#         var_f2 = np.asarray(l_com_2_var).reshape(-1,)
#         mean_f3 = np.asarray(l_com_3_mean).reshape(-1,)
#         var_f3 = np.asarray(l_com_3_var).reshape(-1,)

#         mean_g1 = np.asarray(l_act_1_mean).reshape(-1,)
#         var_g1 = np.asarray(l_act_1_var).reshape(-1,)
#         mean_g2 = np.asarray(l_act_2_mean).reshape(-1,)
#         var_g2 = np.asarray(l_act_2_var).reshape(-1,)
#         mean_g3 = np.asarray(l_act_3_mean).reshape(-1,)
#         var_g3 = np.asarray(l_act_3_var).reshape(-1,)

#         mean_f = [mean_f1, mean_f2, mean_f3]
#         mean_g = [mean_g1, mean_g2, mean_g3]
#         var_f = [var_f1, var_f2, var_f3]
#         var_g = [var_g1, var_g2, var_g3]

#         return mean_f, var_f, mean_g, var_g
    
    
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


























#
