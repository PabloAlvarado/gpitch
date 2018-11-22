import numpy as np
import tensorflow as tf
import gpitch
import gpflow
from gpflow import settings
from gpflow.params import Minibatch
from .likelihoods import MpdLik
from gpflow.params import Parameter, ParamList
from gpflow.kullback_leiblers import gauss_kl
from gpitch.methods import logistic_tf
from gpflow.decors import params_as_tensors
from gpflow import conditionals


float_type = settings.dtypes.float_type
jitter = settings.numerics.jitter_level


def predict_windowed(model, xnew, ws=1600):
    n = xnew.size
    m_a_l = [[] for _ in range(model.num_sources)]
    v_a_l = [[] for _ in range(model.num_sources)]
    m_c_l = [[] for _ in range(model.num_sources)]
    v_c_l = [[] for _ in range(model.num_sources)]
    m_s_l = [[] for _ in range(model.num_sources)]

    for i in range(n/ws):
        x = xnew[i*ws : (i+1)*ws].copy()
        m_a, v_a = model.predict_act(x)
        m_c, v_c = model.predict_com(x)

        for j in range(model.num_sources):
            m_a_l[j].append(m_a[j].copy())
            v_a_l[j].append(v_a[j].copy())
            m_c_l[j].append(m_c[j].copy())
            v_c_l[j].append(v_c[j].copy())
            m_s_l[j].append(gpitch.logistic(m_a[j]) * m_c[j])

    for j in range(model.num_sources):
        m_a_l[j] = np.asarray(m_a_l[j]).reshape(-1, 1)
        v_a_l[j] = np.asarray(v_a_l[j]).reshape(-1, 1)
        m_c_l[j] = np.asarray(m_c_l[j]).reshape(-1, 1)
        v_c_l[j] = np.asarray(v_c_l[j]).reshape(-1, 1)
        m_s_l[j] = gpitch.logistic(m_a_l[j]) * m_c_l[j]

    return m_a_l, v_a_l, m_c_l, v_c_l, m_s_l


class Pdgp(gpflow.models.Model):
    def __init__(self, x, y, z, kern, whiten=True, minibatch_size=None, nlinfun=logistic_tf, **kwargs):
        """
        Pitch detection using Gaussian process.

        Constructor.
        :param x:
        :param y:
        :param z:
        :param kern_com:
        :param kern_act:
        :param transform:
        :param whiten:
        :param minibatch_size:
        """

        gpflow.models.Model.__init__(self, **kwargs)

        if minibatch_size is None:
            minibatch_size = x.shape[0]

        self.minibatch_size = minibatch_size
        self.num_data = x.shape[0]
        self.num_sources = len(kern[0])
        self.whiten = whiten
        self.nlinfun = nlinfun
        self.likelihood = MpdLik(nlinfun=self.nlinfun, num_sources=self.num_sources)

        self.x = Minibatch(x, minibatch_size, np.random.RandomState(0))
        self.y = Minibatch(y, minibatch_size, np.random.RandomState(0))

        self.kern_act = ParamList(kern[0])
        self.kern_com = ParamList(kern[1])

        self.num_inducing_c = []
        self.num_inducing_a = []

        za_l = []
        zc_l = []
        q_mu_com_l = []
        q_mu_act_l = []
        q_sqrt_com_l = []
        q_sqrt_act_l = []

        for i in range(self.num_sources):
            self.num_inducing_a.append(z[0][i].size)
            self.num_inducing_c.append(z[1][i].size)

            za_l.append(Parameter(z[0][i].copy() ))
            zc_l.append(Parameter(z[1][i].copy() ))

            q_mu_act_l.append(Parameter(np.zeros(z[0][i].shape)))
            q_mu_com_l.append(Parameter(np.zeros(z[1][i].shape)))

            q_sqrt_act_l.append(Parameter(np.array([np.eye(self.num_inducing_a[i]) for _ in range(1)]).swapaxes(0, 2)))
            q_sqrt_com_l.append(Parameter(np.array([np.eye(self.num_inducing_c[i]) for _ in range(1)]).swapaxes(0, 2)))

        self.za = ParamList(za_l)
        self.zc = ParamList(zc_l)
        self.q_mu_com = ParamList(q_mu_com_l)
        self.q_mu_act = ParamList(q_mu_act_l)
        self.q_sqrt_com = ParamList(q_sqrt_com_l)
        self.q_sqrt_act = ParamList(q_sqrt_act_l)

    @params_as_tensors
    def build_prior_kl(self):
        """
        compute KL divergences.
        :return:
        """

        if self.whiten:
            kl_act = [gauss_kl(self.q_mu_act[i], self.q_sqrt_act[i]) for i in range(self.num_sources)]
            kl_com = [gauss_kl(self.q_mu_com[i], self.q_sqrt_com[i]) for i in range(self.num_sources)]
        else:
            k_act, k_com = [], []
            kl_act, kl_com = [], []
            for i in range(self.num_sources):
                k_act.append(self.kern_act[i].K(self.za[i]) + tf.eye(self.num_inducing_a[i], dtype=float_type)*jitter)
                k_com.append(self.kern_com[i].K(self.zc[i]) + tf.eye(self.num_inducing_c[i], dtype=float_type)*jitter)
                kl_act.append(gauss_kl(self.q_mu_act[i], self.q_sqrt_act[i], k_act[i]))
                kl_com.append(gauss_kl(self.q_mu_com[i], self.q_sqrt_com[i], k_com[i]))

        return tf.reduce_sum(kl_act) + tf.reduce_sum(kl_com)

    @params_as_tensors
    def _build_likelihood(self):

        # Get prior kl
        kl = self.build_prior_kl()

        # Get conditionals
        fmean, fvar = self._build_predict(self.x, full_cov=False)

        # Get variational expectations
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.x)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - kl

    @params_as_tensors
    def _build_predict(self, xnew, full_cov=False):
        mean_act = self.num_sources * [None]
        mean_com = self.num_sources * [None]
        var_act = self.num_sources * [None]
        var_com = self.num_sources * [None]

        for i in range(self.num_sources):
            mean_act[i], var_act[i] = conditionals.conditional(self.za[i],
                                                           self.kern_act[i],
                                                           xnew,
                                                           self.q_mu_act[i],
                                                           q_sqrt=self.q_sqrt_act[i],
                                                           full_cov=full_cov,
                                                           whiten=self.whiten)

            mean_com[i], var_com[i] = conditionals.conditional(self.zc[i],
                                                           self.kern_com[i],
                                                           xnew,
                                                           self.q_mu_com[i],
                                                           q_sqrt=self.q_sqrt_com[i],
                                                           full_cov=full_cov,
                                                           whiten=self.whiten)

        mean_act_concat = tf.concat(mean_act, 1)
        var_act_concat = tf.concat(var_act, 1)

        mean_com_concat = tf.concat(mean_com, 1)
        var_com_concat = tf.concat(var_com, 1)

        fmean = tf.concat([mean_act_concat, mean_com_concat], 1)
        fvar = tf.concat([var_act_concat, var_com_concat], 1)
        return fmean, fvar

    @params_as_tensors
    def predict_act(self, xnew):
        mean, var = self.num_sources*[None], self.num_sources*[None]
        for i in range(self.num_sources):
            mean[i], var[i] = conditionals.conditional(xnew, self.za[i], self.kern_act[i],
                                                              self.q_mu_act[i], q_sqrt=self.q_sqrt_act[i],
                                                              full_cov=False, whiten=self.whiten)
        return mean, var

    @params_as_tensors
    def predict_com(self, xnew):
        mean, var = self.num_sources*[None], self.num_sources*[None]
        for i in range(self.num_sources):
            mean[i], var[i] = conditionals.conditional(xnew, self.zc[i], self.kern_com[i],
                                                              self.q_mu_com[i], q_sqrt=self.q_sqrt_com[i],
                                                              full_cov=False, whiten=self.whiten)
        return mean, var

    @params_as_tensors
    def predict_act_n_com(self, xnew):

        mean_a, var_a = self.num_sources*[None], self.num_sources*[None]
        mean_c, var_c = self.num_sources*[None], self.num_sources*[None]
        mean_source = self.num_sources*[None]

        for i in range(self.num_sources):
            mean_a[i], var_a[i] = conditionals.conditional(xnew, self.za[i], self.kern_act[i],
                                                                  self.q_mu_act[i], q_sqrt=self.q_sqrt_act[i],
                                                                  full_cov=False, whiten=self.whiten)

            mean_c[i], var_c[i] = conditionals.conditional(xnew, self.zc[i], self.kern_com[i],
                                                                  self.q_mu_com[i], q_sqrt=self.q_sqrt_com[i],
                                                                  full_cov=False, whiten=self.whiten)

            mean_source[i] = self.nlinfun(mean_a[i])*mean_c[i]
        return mean_a, var_a, mean_c, var_c, mean_source
    
    


















#