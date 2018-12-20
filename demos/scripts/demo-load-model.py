import pickle
import matplotlib.pyplot as plt
from gpitch import load_filenames
from gpitch.paths import path_test


test_fnames = load_filenames(directory=path_test,
                             pattern='',
                             pitches=None,
                             ext='.wav')
path_cpu = "/run/user/1000/gvfs/sftp:host=frank.eecs.qmul.ac.uk/import/c4dm-04/alvarado/results/amt/transcription/"
model = pickle.load(open(path_cpu + test_fnames[1].strip(".wav") + "_model.p", "rb"))

model.plot()
plt.show()

