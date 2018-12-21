import sys
import pickle
from gpitch.methods import load_filenames
from gpitch.transcription_svi import AmtSvi
from gpitch.paths import path_train, path_test, path_load, path_save

test_fnames = load_filenames(directory=path_test,
                             pattern='',
                             pitches=None,
                             ext='.wav')

fname = test_fnames[int(sys.argv[1])]  # file to analyze
path = [path_train, path_test, path_load]  # paths to train and test data, learned kernels
frames = [88200, 5 * 44100]  # train and test number of frames

# define model
m = AmtSvi(test_fname=fname, frames=frames, path=path, gpu=sys.argv[2])

# optimization
# maxiter = int(sys.argv[3])
m.optimize(maxiter=100000, learning_rate=0.01)

# prediction
m.predict()

# save model
save_name = path_save + fname.strip(".wav") + "_model.p"
pickle.dump(obj=m,
            file=open(save_name, "wb"))
