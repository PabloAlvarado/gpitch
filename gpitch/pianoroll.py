import numpy as np
import gpitch
import pandas as pd


class Pianoroll:

    def __init__(self, path=None, filename=None, fs=100, duration=10.):
        self.filename = filename
        self.path = path
        self.duration = duration
        self.fs = fs
        self.xn = int(round(duration * fs))
        self.x = np.linspace(0., (self.xn - 1.) / self.fs, self.xn).reshape(-1, 1)
        self.pitch_range = range(21, 109)
        self.midi_dict = self.init_dict()
        self.per_dict = self.init_dict()
        self.name = "name"
        self.pitch_list = []

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
        pandas_file = pandas_file[idx]
        self.pitch_list = list(set(pandas_file.MidiPitch.tolist()))
        self.pitch_list.sort()

        for i in range(len(self.pitch_list)):

            pitch_pandas = pandas_file[pandas_file.MidiPitch == self.pitch_list[i]]
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

    def compute_periodogram(self, binarize=False, th=0.1):
        per = []  # periodogram

        for pitch in self.pitch_range:
            per.append(self.per_dict[str(pitch)].copy())

        per = np.asarray(per).reshape(len(self.pitch_range), -1)
        per = np.flipud(per)

        if binarize:
            per[per < th] = 0.
            per[per >= th] = 1.
        return per

    def mir_eval_format(self):

        # detect onsets
        # detect offsets
        # convert midi to Hz
        # export arrays for mir_eval
        pass
