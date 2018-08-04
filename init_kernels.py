from gpflow.kernels import Matern32
from gpitch.kernels import MercerCosMix
import gpflow


def init_kern_act(num_pitches):
    """Initialize kernels for activations and components"""

    kern_act = []

    for i in range(num_pitches):
        kern_act.append(Matern32(1, lengthscales=1.0, variance=3.5))
    return kern_act


def init_kern_com(num_pitches, lengthscale, energy, frequency):
    """Initialize kernels for activations and components"""

    kern_com, kern_exp, kern_per = [], [], []

    for i in range(num_pitches):
        kern_exp.append(Matern32(1, lengthscales=lengthscale[i], variance=1.0) )
        #kern_exp[i].variance.fixed = True
        #kern_exp[i].lengthscales.fixed = True
        kern_exp[i].lengthscales.transform = gpflow.transforms.Logistic(0., 1.)

        kern_per.append(MercerCosMix(1, energy=energy[i], frequency=frequency[i], variance=1.0,))
        kern_per[i].fixed = True

        kern_com.append( kern_exp[i] * kern_per[i] )
    return kern_com


def init_kern(num_pitches, lengthscale, energy, frequency):
    """Initialize kernels for activations and components"""

    kern_act = init_kern_act(num_pitches)
    kern_com = init_kern_com(num_pitches, lengthscale, energy, frequency)
    kern = [kern_act, kern_com]
    return kern