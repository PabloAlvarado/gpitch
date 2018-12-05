import pickle
import gpitch
from gpitch.models import GpitchModel
from gpflow.kernels import Matern32
from gpitch.matern12_spectral_mixture import MercerMatern12sm as Mercer
import tensorflow as tf
import matplotlib.pyplot as plt
from gpitch.myplots import plot_predict
import numpy as np
import time


class Logger:
    def __init__(self, model):
        self.model = model
        self.i = 1
        self.logf = []

    def callback(self, x):
        if (self.i % 20) == 0:
            self.logf.append(self.model._objective(x)[0])
        self.i += 1

    def array(self):
        return (-np.array(self.logf))


class AmtSvi(GpitchModel):
    """
    Automatic music transcription using stochastic variational inference class
    """
    def __init__(self, test_fname, frames, path, pitches=None, gpu='0', maps=True, extrema=True,
                 minibatch_size=100, reg=False):
        GpitchModel.__init__(self,
                             pitches=pitches,
                             test_fname=test_fname,
                             frames=frames,
                             path=path,
                             gpu=gpu,
                             maps=maps,
                             extrema=extrema)
        self.reg = reg
        self.path_load = path[2]
        self.kernels = self.init_kernels()
        self.model = self.init_model(minibatch_size)
        self.logger = Logger(self.model)
        self.prediction = None

    def plot_results(self, figsize=(12, 2 * 88)):
        """
        plot prediction components and activations
        """
        plt.figure(figsize=figsize)
        m_a, v_a, m_c, v_c, esource = self.prediction
        for j in range(len(self.pitches)):
            plt.subplot(88, 2, 2 * (j + 1) - 1)
            plot_predict(self.data_test.x.copy(),
                         m_a[j],
                         v_a[j],
                         self.model.za[j].value,
                         plot_z=False,
                         latent=True,
                         plot_latent=False)

            plt.subplot(88, 2, 2 * (j + 1))
            plot_predict(self.data_test.x.copy(),
                         m_c[j],
                         v_c[j],
                         self.model.zc[j].value,
                         plot_z=False)

        plt.figure(figsize=figsize)
        for j in range(len(self.pitches)):
            plt.subplot(88, 1, j+1)
            plt.plot(self.data_test.x, self.data_test.y)
            plt.plot(self.data_test.x, esource[j])
            plt.plot(self.piano_roll.x, self.piano_roll.pr_dic[str(self.pitches[j])], lw=2)

    def predict(self, xnew=None):
        if xnew is None:
            xnew = self.data_test.x.copy()
        self.prediction = self.model.predict_windowed(xnew)

    def optimize(self, maxiter, learning_rate):
        method = tf.train.AdamOptimizer(learning_rate=learning_rate)
        start_time = time.time()
        self.model.optimize(maxiter=maxiter, method=method, callback=self.logger.callback)
        print("Time optimizing (seconds): {0}".format(time.time() - start_time))

    def init_model(self, minibatch_size):
        return gpitch.pdgp.Pdgp(x=self.data_test.x.copy(),
                                y=self.data_test.y.copy(),
                                z=self.z,
                                kern=self.kernels,
                                minibatch_size=minibatch_size,
                                reg=self.reg)

    def init_kernels(self, fixed=True):
        fname = gpitch.load_filenames(directory=self.path_load,
                                      pattern='params',
                                      pitches=self.pitches,
                                      ext='.p')
        params = []
        k_act, k_com = [], []
        for i in range(len(fname)):

            k_act.append(Matern32(1, lengthscales=0.2, variance=3.5))

            params.append(
                pickle.load(
                    open(self.path_load + fname[i], "rb")
                )
            )

            k_com.append(
                         Mercer(input_dim=1,
                                energy=params[i][1],
                                frequency=params[i][2],
                                lengthscales=params[i][0],
                                variance=1.)
            )
            if fixed:
                k_com[i].energy.fixed = True
                k_com[i].frequency.fixed = True
                k_com[i].lengthscales.fixed = True

        return [k_act, k_com]
