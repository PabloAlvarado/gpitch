from gpitch.transcription_svi import AmtSvi
from gpitch.paths import path_train, path_test, path_load
import matplotlib.pyplot as plt
import gpitch

path_cpu = "/run/user/1000/gvfs/sftp:host=frank.eecs.qmul.ac.uk"
path = [
    path_cpu + path_train,
    path_cpu + path_test,
    path_cpu + path_load]  # paths to train and test data, learned kernels

test_fnames = gpitch.load_filenames(directory=path[1],
                                    pattern='',
                                    pitches=None,
                                    ext='.wav')

fname = test_fnames[1]  # file to analyze
frames = [88200, 2 * 44100]  # train and test number of frames

# define model
m = AmtSvi(test_fname=fname, frames=frames, path=path)

# optimization
m.optimize(maxiter=2500, learning_rate=0.01)

# prediction
m.predict()

# get list onsets, offset, pitches
ref_pitches, ref_intervals = m.piano_roll.mir_eval_format(ground_truth=True)
est_pitches, est_intervals = m.prediction_pr.mir_eval_format()

# compute metrics
metrics = gpitch.compute_mir_eval(ref_pitches=ref_pitches,
                                  ref_intervals=ref_intervals,
                                  est_pitches=est_pitches,
                                  est_intervals=est_intervals,
                                  offset_ratio=None)
print metrics

# show model
m.plot()
plt.show()
