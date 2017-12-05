import sys
import numpy as np
import tensorflow as tf
import pickle
import gpflow
import soundfile
sys.path.append('../../')
import gpitch


visible_device = sys.argv[1]  # gpu to use
namefile = sys.argv[2].strip('.wav')  # file to analize
dataloc = '../../../datasets/yoshii_data/training_data/'  # location of training data
audioname = dataloc + namefile + '.wav'  # load data
gpitch.amtgp.init_settings(visible_device=visible_device, interactive=False)  # set gpu

print('Training using file ' + namefile + ' and GPU ' + visible_device)
N = 16000  # number of frames
dec = 160  # decimation factor
minibatch_size = 200  # batch size svi
maxiter = 2000  #  maximun number of iterations in optimization
learning_rate = 0.01  # learning rate svi optimization
maxh = 20  # max number of harmoncis in component kernel
x, y, fs = gpitch.amtgp.readaudio(audioname, frames=N, start=0)  # load data
ideal_f0 = gpitch.amtgp.find_ideal_f0(namefile)  # get ideal f0 from file name
icom = gpitch.amtgp.init_com_params(y=y, fs=fs, maxh=maxh, ideal_f0=ideal_f0, scaled=True,
                                    win_size=10)  # init component parameters
Nc = icom[0].size  # number of harmonics per component
ker_com = gpitch.kernels.MaternSpecMix(input_dim=1, lengthscales=0.1, frequencies=icom[0],
                                       variances=icom[1], Nc=Nc)  # define comp kernel
ker_act = gpflow.kernels.Matern32(input_dim=1, lengthscales=1., variance=10.)  # def act k
z = np.vstack((x[::dec].copy(), x[-1].copy()))
m = gpitch.modgp.ModGP(x=x, y=y, z=z, kern_com=ker_com, kern_act=ker_act, whiten=True,
                       minibatch_size=minibatch_size)
m.kern_com.fixed = True # Set all parameters free to optimize, but variances of component
m.kern_com.lengthscales.fixed = False
m.kern_com.lengthscales.transform = gpflow.transforms.Logistic(0., 10.0)
m.fixed_msmkern_params(freq=False, var=True)
m.kern_act.fixed = False
m.likelihood.variance.fixed = False
m.z.fixed = True
m.optimize_svi(maxiter=maxiter, learning_rate=learning_rate)  # optimize
W = 10
Nw = N / W
xnew = [x[Nw*i : Nw*(i+1)].copy() for i in range(W)]
prediction = m.predict_all(xnew)  # predict
m.prediction_save = prediction
pickle.dump(m, open('/import/c4dm-04/alvarado/results/ss_amt/trained_model_'
                    + namefile + ".p", "wb"))
