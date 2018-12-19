import numpy as np
import gpitch
import pandas as pd


class Pianoroll:

    def __init__(self, x, path=None, filename=None, fs=44100, threshold=0.05):
        self.threshold = threshold
        self.filename = filename
        self.path = path
        self.fs = fs
        self.x = x
        self.xn = x.size
        self.duration = x[-1, 0].copy()
        self.pitch_range = range(21, 109)
        self.midi_dict = self.init_dict()
        self.per_dict = self.init_dict()
        self.name = "name"
        self.pitch_list = None
        self.pandas_file = None

        if self.filename is not None:
            self.load_file()

    def init_dict(self):
        plist = []  # init dictionaries

        for i in self.pitch_range:
            plist.append(
                (str(i),
                 np.zeros((self.xn, 1))
                 )
            )
        return dict(plist)

    def load_file(self):
        self.name = gpitch.load_filenames(directory=self.path,
                                          pattern=self.filename.strip('.wav'),
                                          pitches=None,
                                          ext='.txt')[0]

        pandas_file = pd.read_table(self.path + self.name)
        idx = pandas_file["OnsetTime"] < self.duration
        self.pandas_file = pandas_file[idx]
        self.pitch_list = list(set(self.pandas_file.MidiPitch.tolist()))
        self.pitch_list.sort()

        for i in range(len(self.pitch_list)):

            pitch_pandas = self.pandas_file[self.pandas_file.MidiPitch == self.pitch_list[i]]
            onset = pitch_pandas.OnsetTime.tolist()
            offset = pitch_pandas.OffsetTime.tolist()
            key = str(self.pitch_list[i])

            for j in range(len(onset)):
                self.midi_dict[key][(onset[j] <= self.x) & (self.x < offset[j])] = 1.

    def compute_midi(self):
        midi = []

        for pitch in self.pitch_range:
            midi.append(self.midi_dict[str(pitch)].copy())

        midi = np.asarray(midi).reshape(len(self.pitch_range), -1)
        midi = np.flipud(midi)
        return midi

    def compute_periodogram(self, binarize=False):
        per = []  # periodogram

        for pitch in self.pitch_range:
            per.append(self.per_dict[str(pitch)].copy())

        per = np.asarray(per).reshape(len(self.pitch_range), -1)
        per = np.flipud(per)

        if binarize:
            per = self.binarize(per)
        return per

    def binarize(self, pr):
        pr[pr < self.threshold] = 0.
        pr[pr >= self.threshold] = 1.
        return pr

    def get_array_pandas(self, key):
        list_ = self.pandas_file[key].tolist()
        return np.asarray(list_).reshape(-1, )

    def mir_eval_format(self, ground_truth=False):

        if ground_truth:

            onsets = self.get_array_pandas("OnsetTime")
            offsets = self.get_array_pandas("OffsetTime")
            intervals = np.ones((onsets.size, 2))
            intervals[:, 0] = onsets.copy()
            intervals[:, 1] = offsets.copy()
            pitches_midi = self.get_array_pandas("MidiPitch")
            pitches = gpitch.midi2freq(pitches_midi)

        else:
            list_onsets = []
            list_pitches = []
            for pitch in self.pitch_list:

                envelope = self.per_dict[str(pitch)].copy()
                gate = self.binarize(envelope.copy())
                diff = np.diff(gate.reshape(-1, ))
                sign = np.sign(diff)
                onsets = sign.copy()
                onsets[onsets < 0] = 0.
                idx_onsets = np.argwhere(onsets)
                found_onsets = self.x[idx_onsets, 0]  # detect onsets

                # offsets = sign.copy()
                # offsets[offsets > 0] = 0.
                # offsets = np.abs(offsets)
                # idx_offsets = np.argwhere(offsets)
                # list_offsets = self.x[idx_offsets, 0]  # detect offsets

                if found_onsets.size is not 0:
                    list_onsets.append(found_onsets)
                    list_pitches.append(found_onsets.size * [gpitch.midi2freq(pitch)])

            array_onsets = np.vstack(list_onsets).reshape(-1, )
            array_pitches = np.hstack(list_pitches).reshape(-1, )

            # sort by onset time
            idx = np.argsort(array_onsets)
            array_onsets = array_onsets[idx]
            array_pitches = array_pitches[idx]

            intervals = np.ones((array_onsets.size, 2))
            intervals[:, 0] = array_onsets.copy()
            intervals[:, 1] = 1. + array_onsets.copy()
            pitches = array_pitches

        return pitches, intervals
