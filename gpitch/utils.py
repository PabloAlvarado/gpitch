import gpitch
import pickle
from gpitch.matern12_spectral_mixture as Mercer


def init_kern_component(path, pitches, fixed=True):
    fname = gpitch.load_filenames(directory=path,
                                  pattern='params',
                                  pitches=pitches,
                                  ext='.p')
    params = []
    kernels = []
    for i in range(len(fname)):
        params.append(
                      pickle.load(
                                  open(path + fname[i], "rb")
                                 )
                     )

        kernels.append(
                       Mercer(input_dim=1,
                              energy=params[i][1],
                              frequency=params[i][2],
                              lengthscales=params[i][0],
                              variance=1.)
                      )
        if fixed:
            kernels[i].energy.fixed = True
            kernels[i].frequency.fixed = True
            kernels[i].lengthscales.fixed = True

    return kernels