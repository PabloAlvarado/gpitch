from gpitch import init_settings, load_filenames
from gpitch.init_models import init_inducing, init_inducing_extrema
from gpitch.audio import Audio
import matplotlib.pyplot as plt
import numpy as np
from gpitch.pianoroll import Pianoroll


class GpitchModel:
    def __init__(self, test_fname, frames, path, pitches=None, gpu='0',
                 maps=True, extrema=True, start=0):

        self.path_train = path[0]
        self.path_test = path[1]
        self.data_test = Audio(path=self.path_test, filename=test_fname, frames=frames[1],
                               scaled=True, start=start)
        init_settings(gpu)


        # create piano roll object
        if maps:
            self.piano_roll = Pianoroll(path=self.path_test,
                                        filename=test_fname,
                                        duration=self.data_test.x[-1, 0].copy())

        if pitches is None:
            self.pitches = self.piano_roll.pitch_list
        else:
            self.pitches = pitches

        self.data_train = self.load_train_data(maps, frames[0])

        if extrema:
            self.z, self.u_init = self.init_inducing(extrema)
        else:
            self.z = self.init_inducing(extrema)

    def init_inducing(self, extrema, nivps=50):
        if extrema:
            z, u = init_inducing_extrema(x=self.data_test.x.copy(),
                                         y=self.data_test.y.copy(),
                                         num_sources=len(self.pitches),
                                         win_size=37,
                                         thres=0.05,
                                         dec=29)
            return z, u
        else:
            z = init_inducing(x=self.data_test.x.copy(),
                              num_sources=len(self.pitches),
                              nivps_a=nivps,
                              nivps_c=nivps,
                              fs=self.data_test.fs)
            return z

    def load_train_data(self, maps, frames):

        if maps:
            pattern = 'F'
        else:
            pattern = ''
        fnames = load_filenames(directory=self.path_train,
                                pattern=pattern,
                                pitches=self.pitches,
                                ext=".wav")

        data = []
        for p in fnames:
            if p.find("S1") is not -1:
                start = 30000
            else:
                start = 20000
            data.append(
                        Audio(path=self.path_train,
                              filename=p,
                              start=start,
                              frames=frames
                              )
                       )
        return data

    def plot_data_test(self, xlim=None, figsize=None):

        if figsize is None:
            figsize = (12, 2)

        if xlim is None:
            xlim = (self.data_test.x[0], self.data_test.x[-1])

        plt.figure(figsize=figsize)
        plt.plot(self.data_test.x, self.data_test.y)
        plt.plot(self.z[0][0], self.u_init, '|', mew=2)
        plt.xlim(xlim)
        plt.title("test data")

    def plot_data_train(self, figsize=None, axis_off=True):
        nfiles = len(self.data_train)

        if nfiles <= 4:
            ncols = nfiles
        else:
            ncols = 4

        nrows = int(np.ceil(nfiles / 4.))

        if figsize is None:
            figsize = (12, 2 * nrows)

        plt.figure(figsize=figsize)
        for i in range(nfiles):
            plt.subplot(nrows, ncols, i + 1)
            plt.plot(self.data_train[i].x, self.data_train[i].y)
            plt.legend([self.data_train[i].filename[18:-13]])
            if axis_off:
                plt.axis("off")
        plt.suptitle("train data")

