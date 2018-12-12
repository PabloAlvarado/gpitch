import matplotlib.pyplot as plt
from gpitch.transcription_svi import AmtSvi


#path = '/run/user/1000/gvfs/sftp:host=frank.eecs.qmul.ac.uk'
path = ''
path_test = path + '/import/c4dm-01/MAPS_original/AkPnBcht/MUS/'
path_train = path + '/import/c4dm-01/MAPS_original/AkPnBcht/ISOL/NO/'
path_load = path + '/import/c4dm-04/alvarado/results/amt/params/'
fname = 'MAPS_MUS-bach_847_AkPnBcht.wav'
path = [path_train, path_test, path_load]
frames = [88200, 2*44100]  # train and test number of frames

m = AmtSvi(test_fname=fname, frames=frames, path=path)
#m.optimize(maxiter=1000, learning_rate=0.01)
m.predict()
m.predict_pianoroll()

m.plot_data_train()
m.plot_data_test()
m.plot_results()
plt.show()
