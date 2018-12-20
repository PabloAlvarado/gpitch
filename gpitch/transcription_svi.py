import pickle
import gpitch
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import gpflow
from gpitch.models import GpitchModel
from gpitch.matern12_spectral_mixture import MercerMatern12sm as Mercer
from gpitch.matern12_spectral_mixture import Matern12sm
from scipy import signal
from gpitch.myplots import plot_predict
from gpitch.pianoroll import Pianoroll


# callback to get elbo values
class Logger:
    def __init__(self, model):
        self.model = model
        self.i = 1
        self.logf = []

    # noinspection PyProtectedMember
    def callback(self, x):
        if (self.i % 20) == 0:
            self.logf.append(self.model._objective(x)[0])
        self.i += 1

    def array(self):
        return -np.array(self.logf)


class AmtSvi(GpitchModel):
    """
    Automatic music transcription using stochastic variational inference class
    """

    def __init__(self, test_fname,
                 frames, path,
                 pitches=None, gpu='0',
                 maps=True, extrema=True,
                 minibatch_size=100, reg=False,
                 mercer=True, start=0):
        GpitchModel.__init__(self,
                             pitches=pitches,
                             test_fname=test_fname,
                             frames=frames,
                             path=path,
                             gpu=gpu,
                             maps=maps,
                             extrema=extrema,
                             start=start)
        self.reg = reg
        self.mercer = mercer
        self.path_load = path[2]
        self.kernels = self.init_kernels()
        self.model = self.init_model(minibatch_size)
        self.logger = Logger(self.model)
        self.prediction = None
        self.pitch_dim = len(self.pitches)

        self.prediction_pr = Pianoroll(path=self.path_test,
                                       filename=test_fname,
                                       x=self.data_test.x.copy())

        self.model.za.fixed = True
        self.model.zc.fixed = True
        print("analyzing file {0}".format(test_fname))
        print("number of inducing variables: {0}".format(len(self.model.za[0].value)))
        print("pitches to detect {0}".format(self.pitches))

    def compute_metrics(self):
        # get list onsets, offset, pitches
        ref_pitches, ref_intervals = self.piano_roll.mir_eval_format(ground_truth=True)
        est_pitches, est_intervals = self.prediction_pr.mir_eval_format()

        # compute metrics
        metrics = gpitch.compute_mir_eval(ref_pitches=ref_pitches,
                                          ref_intervals=ref_intervals,
                                          est_pitches=est_pitches,
                                          est_intervals=est_intervals,
                                          offset_ratio=None)
        return metrics

    def save_results(self, save_path):
        aux = save_path + self.data_test.filename.strip(".wav") + "_transcription.p"
        pickle.dump(
            obj=self.prediction,
            file=open(
                aux, "wb"
            ),
            protocol=2
        )

    def plot(self):
        self.plot_data_train()
        self.plot_data_test()
        self.plot_results()

    def plot_results(self, figsize=None):
        """
        plot prediction components and activations
        """

        # plot activations and components
        if figsize is None:

            figsize = (12, 2 * self.pitch_dim)

        plt.figure(figsize=figsize)
        m_a, v_a, m_c, v_c, esource = self.prediction

        for j in range(len(self.pitches)):

            plt.subplot(self.pitch_dim, 2, 2 * (j + 1) - 1)
            plot_predict(self.data_test.x.copy(),
                         m_a[j],
                         v_a[j],
                         self.model.za[j].value,
                         plot_z=False,
                         latent=True,
                         plot_latent=False)

            plt.subplot(self.pitch_dim, 2, 2 * (j + 1))
            plot_predict(self.data_test.x.copy(),
                         m_c[j],
                         v_c[j],
                         self.model.zc[j].value,
                         plot_z=False)

        # plot sources
        plt.figure(figsize=figsize)

        for j in range(len(self.pitches)):

            plt.subplot(self.pitch_dim, 1, j + 1)
            plt.plot(self.data_test.x, self.data_test.y)
            plt.plot(self.data_test.x, esource[j])
            plt.plot(self.piano_roll.x, self.piano_roll.midi_dict[str(self.pitches[j])], lw=2)
            plt.plot(self.prediction_pr.x, self.prediction_pr.per_dict[str(self.pitches[j])], lw=2)

        # plot pianoroll ground truth
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(self.piano_roll.compute_midi(),
                   cmap=plt.cm.get_cmap('binary'),
                   interpolation="none",
                   extent=[self.data_test.x[0], self.data_test.x[-1], 21, 108],
                   aspect="auto")
        freq, inter = self.piano_roll.mir_eval_format(ground_truth=True)
        midi = gpitch.freq2midi(freq)
        onsets = inter[:, 0]
        plt.plot(onsets, midi, 'sr', ms=4, mfc='none', mew=1)
        plt.title("ground truth")

        # plot pianoroll prediction
        plt.subplot(1, 3, 2)
        plt.imshow(self.prediction_pr.compute_periodogram(binarize=True),
                   cmap=plt.cm.get_cmap('binary'),
                   interpolation="none",
                   extent=[self.data_test.x[0], self.data_test.x[-1], 21, 108],
                   aspect="auto")
        freq, inter = self.prediction_pr.mir_eval_format()
        midi = gpitch.freq2midi(freq)
        onsets = inter[:, 0]
        plt.plot(onsets, midi, 'sr', ms=4, mfc='none', mew=1)
        plt.title("prediction")

        # plot ELBO
        plt.subplot(1, 3, 3)
        plt.title("ELBO")
        plt.plot(self.logger.array())

        # plot envelopes
        plt.figure(figsize=(12, 4))
        plt.title("envelopes")
        for key in self.prediction_pr.pitch_list:
            plt.plot(self.prediction_pr.x, self.prediction_pr.per_dict[str(key)])
        plt.show()

    def predict_pianoroll(self):

        win = signal.hann(2205)  # smoothing window
        aux1 = []
        aux2 = []
        for i in range(self.pitch_dim):
            # get absolute value sources
            aux1.append(np.abs(self.prediction[-1][i]))

            # get envelope
            aux2.append(signal.convolve(aux1[i].reshape(-1), win, mode='same') / win.size)
            aux2[i] = np.max(aux1[i]) * aux2[i] / np.max(aux2[i])

            # downsample
            # aux2[i] = aux2[i][::44].reshape(-1, 1)
            aux2[i] = aux2[i][::1].reshape(-1, 1)

            # save on periodogram dictionary
            self.prediction_pr.per_dict[str(self.pitches[i])] = aux2[i]

    def predict(self, xnew=None):
        if xnew is None:
            xnew = self.data_test.x.copy()
        self.prediction = self.model.predict_windowed(xnew)
        self.predict_pianoroll()

    def optimize(self, maxiter, learning_rate):
        method = tf.train.AdamOptimizer(learning_rate=learning_rate)
        start_time = time.time()
        self.model.optimize(maxiter=maxiter, method=method, callback=self.logger.callback)
        print("Time optimizing (minutes): {0}".format((time.time() - start_time) / 60.))

    def init_model(self, minibatch_size):
        return gpitch.pdgp.Pdgp(x=self.data_test.x.copy(),
                                y=self.data_test.y.copy(),
                                z=self.z,
                                kern=self.kernels,
                                minibatch_size=minibatch_size,
                                reg=self.reg)

    def init_kernels(self, fixed=True, maxh=20):
        fname = gpitch.load_filenames(directory=self.path_load,
                                      pattern='params',
                                      pitches=self.pitches,
                                      ext='.p')
        params = []
        k_act, k_com = [], []
        for i in range(len(fname)):

            k_act.append(gpflow.kernels.Matern32(1, lengthscales=1.0, variance=3.5))

            params.append(
                pickle.load(
                    open(self.path_load + fname[i], "rb")
                )
            )

            if self.mercer:
                k_com.append(
                    Mercer(input_dim=1,
                           energy=params[i][1][0:maxh],
                           frequency=params[i][2][0:maxh],
                           lengthscales=params[i][0],
                           variance=1.
                           )
                )
            else:
                k_com.append(
                    Matern12sm(input_dim=1,
                               energy=params[i][1][0:maxh],
                               frequency=params[i][2][0:maxh],
                               lengthscales=params[i][0],
                               variance=1.)
                )
            if fixed:
                k_com[i].energy.fixed = True
                k_com[i].frequency.fixed = True
                k_com[i].lengthscales.fixed = True

        return [k_act, k_com]
