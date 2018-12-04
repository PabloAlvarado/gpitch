from gpitch import readaudio, segmented
from gpitch import window_overlap


class Audio:
    def __init__(self, path, filename, frames=-1, start=0, scaled=False, window_size=None,
                 overlap=True, windowed=False):
        self.path = path
        self.filename = filename
        self.x, self.y, self.fs = self.read(frames, start, scaled)

        if windowed:
            if window_size is None:
                window_size = self.x.size
            self.wsize = window_size
            self.X, self.Y = self.windowed(overlap)

    def read(self, frames=-1, start=0, scaled=False):
        return readaudio(fname=self.path + self.filename, frames=frames, start=start,
                         scaled=scaled)

    def windowed(self, overlap):
        if overlap:
            xwin, ywin = window_overlap.windowed(x=self.x, y=self.y, ws=self.wsize)
        else:
            xwin, ywin = segmented(x=self.x, y=self.y, window_size=self.wsize)

        self.X, self.Y = xwin, ywin
        return xwin, ywin
