import numpy as np
import scipy as sp
from scipy import fftpack
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
import GPflow
import time
import gpitch as gpi
import loogp
import sounddevice as sod #reproduce audio from numpy arrays
import soundfile as sof # package to load wav files
reload(loogp)
reload(gpi)




plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
plt.interactive(True)
plt.close('all')



















#
