import gpitch3 as gpitch
import pickle


path = "/import/c4dm-04/alvarado/results/amt/params/"
gpitch.load_filenames
pitches = range(21, 21+88)

fname = gpitch.load_filenames(directory=path,
                                  pattern='params',
                                  pitches=pitches,
                                  ext='.p')

for p in fname:
    params = pickle.load(open(path + p, "rb"))
    print(p)
    pickle.dump(params, open(path + p, "wb"), protocol=2)