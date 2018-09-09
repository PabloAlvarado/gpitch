import numpy as np
import matplotlib.pyplot as plt
import gpflow
import gpitch


class Audio:
    def __init__(self, path=None, filename=None, frames=-1, start=0, scaled=True, window_size=None):

        if path is None:
            self.name = 'unnamed'
            self.fs = 44100
            self.x = np.linspace(0., (self.fs - 1.)/self.fs,  self.fs).reshape(-1, 1)
            self.y = np.cos(2*np.pi*self.x*440.)

        else:
            self.name = filename
            self.x, self.y, self.fs = gpitch.readaudio(fname=path + filename, frames=frames, start=start, scaled=scaled)

        if window_size is None:
            window_size = self.x.size
        self.wsize = window_size
        self.X, self.Y = gpitch.segmented(x=self.x, y=self.y, window_size=window_size)


class AMT:
    """
    Automatic music transcription class
    """

    def __init__(self, test_filename, nsec, pitches, window_size=4410, run_on_server=False, gpu='0'):

        self.train_data = [None]
        self.test_data = Audio()
        self.params = [[], [], []]
        self.kern_sampled = [None]
        self.inducing = [None]
        self.kern_pitches = [None]
        self.model = None

        # init session
        self.sess, self.path = gpitch.init_settings(visible_device=gpu, run_on_server=run_on_server)

        # data_path = "c4dm-01/MAPS_original/AkPnBcht/ISOL/NO/"
        # save_path = "c4dm-04/alvarado/results/"
        self.train_path = "/media/pa/TOSHIBA EXT/Datasets/MAPS/AkPnBcht/ISOL/NO/"
        self.test_path = "/media/pa/TOSHIBA EXT/Datasets/MAPS/AkPnBcht/MUS/"

        self.load_train(pitches=pitches)
        self.load_test(filename=test_filename, start=20000, frames=nsec*44100, window_size=window_size)

    def load_train(self, pitches, train_data_path=None):

        if train_data_path is not None:
            self.train_path = train_data_path

        path = self.path + self.train_path
        lfiles = gpitch.methods.load_filenames(directory=path, pattern='F', pitches=pitches)
        nfiles = len(lfiles)
        data = []

        for i in range(nfiles):
            if lfiles[i].find("S1") is not -1:
                start = 30000
            else:
                start = 20000
            data.append(Audio(path=path, filename=lfiles[i], start=start, frames=88200))
        self.train_data = data

    def load_test(self, filename, window_size, start, frames, train_data_path=None):
        if train_data_path is not None:
            self.test_path = train_data_path
        path = self.path + self.test_path
        self.test_data = Audio(path=path, filename=filename, start=start, frames=frames, window_size=window_size)

    def plot_traindata(self, figsize=None, axis_off=True):
        nfiles = len(self.train_data)

        if nfiles <= 4:
            ncols = nfiles
        else:
            ncols = 4

        nrows = int(np.ceil(nfiles/4.))

        if figsize == None:
            figsize = (16, 2*nrows)

        plt.figure(figsize=figsize)
        for i in range(nfiles):
            plt.subplot(nrows, ncols, i+1)
            plt.plot(self.train_data[i].x, self.train_data[i].y)
            plt.legend([self.train_data[i].name[18:-13]])
            if axis_off:
                plt.axis("off")
        plt.suptitle("train data")

    def plot_testdata(self, figsize=(16, 2), axis_off=True):
        plt.figure(figsize=figsize)
        plt.plot(self.test_data.x, self.test_data.y)
        plt.legend([self.test_data.name])
        plt.title("test data")
        if axis_off:
            plt.axis("off")

    def plot_kernels(self, figsize=None, axis=False):
        nfiles = len(self.train_data)

        if nfiles <= 2:
            ncols = nfiles
        else:
            ncols = 2

        nrows = int(np.ceil(nfiles / 2.))
        x0 = np.array(0.).reshape(-1, 1)

        if figsize == None:
            figsize = (16, 2*nrows)

        plt.figure(figsize=figsize)
        for i in range(nfiles):
            plt.subplot(nrows, ncols, i + 1)

            plt.plot(self.kern_sampled[0][i], self.kern_sampled[1][i])
            plt.plot(self.kern_sampled[0][i], self.kern_pitches[i].compute_K(self.kern_sampled[0][i], x0))
            plt.title(self.train_data[i].name[18:-13])
            plt.legend(['sampled kernel', 'approximate kernel'])
            if axis is not True:
                plt.axis("off")
        plt.suptitle("sampled kernels")

    def load_kernels(self):
        pass

    def init_kernels(self, covsize, num_sam, max_par=20, train=True):

        nfiles = len(self.train_data)
        skern, xkern = nfiles * [None], nfiles * [None]
        scov, samples = nfiles * [None], nfiles * [None]
        if train:
            for i in range(nfiles):

                # sample cov matrix
                scov[i], skern[i], samples[i] = gpitch.samplecov.get_cov(self.train_data[i].y, num_sam=num_sam, size=covsize)

                # approx kernel
                [params, kern_in, kern_fi] = gpitch.kernelfit.fit(kern=skern[i], audio=self.train_data[i].y,
                                                                  file_name=self.train_data[i].name, max_par=max_par)
                self.params[0].append(params[0])  # lengthscale
                self.params[1].append(params[1])  # variances
                self.params[2].append(params[2])   # frequencies

                xkern[i] = np.linspace(0., (covsize - 1.) / self.train_data[i].fs, covsize).reshape(-1, 1)
            self.kern_sampled = [xkern, skern]
            self.kern_pitches = gpitch.init_kernels.init_kern_com(num_pitches=len(self.train_data),
                                                                  lengthscale=self.params[0],
                                                                  energy=self.params[1],
                                                                  frequency=self.params[2])
        else:
            # load already learned parameters
            pass

    def init_inducing(self):
        nwin = len(self.test_data.X)
        u = nwin * [None]
        z = nwin * [None]

        for i in range(nwin):
            a, b = gpitch.init_liv(x=self.test_data.X[i], y=self.test_data.Y[i], num_sources=1)
            z[i] = a[0][0]
            u[i] = b
        self.inducing = [z, u]

    def init_model(self):
        """Hi"""
        self.init_inducing()  # init inducing points

        # init model kernel
        kern_model = np.sum(self.kern_pitches)

        # init gp model
        x_init = self.test_data.X[0].copy()
        y_init = self.test_data.Y[0].copy()
        z_init = self.inducing[0][0].copy()
        self.model = gpflow.sgpr.SGPR(X=x_init, Y=y_init, kern=kern_model, Z=z_init)

    def reset_model(self, x, y, z):
        self.model.X = x
        self.model.Y = y
        self.model.Z = z

    def optimize(self, maxiter, disp=1, nwin=None):

        if nwin is None:
            nwin = len(self.test_data.Y)

        for i in range(nwin):
            if i == 0:
                self.model.optimize(disp=disp, maxiter=maxiter)

            else:
                x_init = self.test_data.X[i].copy()
                y_init = self.test_data.Y[i].copy()
                z_init = self.inducing[0][i].copy()
                self.reset_model(x=x_init, y=y_init, z=z_init)
                self.model.optimize(disp=disp, maxiter=maxiter)

    def save(self):
        pass
