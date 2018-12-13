from gpitch import load_filenames


path_test = '/import/c4dm-01/MAPS_original/AkPnBcht/MUS/'
path_train = '/import/c4dm-01/MAPS_original/AkPnBcht/ISOL/NO/'
path_load = '/import/c4dm-04/alvarado/results/amt/params/'
path_save =  '/import/c4dm-04/alvarado/results/amt/transcription/'
test_fnames = load_filenames(directory=path_test,
                             pattern='',
                             pitches=None,
                             ext='.wav')
