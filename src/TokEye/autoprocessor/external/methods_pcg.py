import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, opening, disk, footprint_rectangle

class AzaSpectral():
    def __init__(self):
        self.quantfilt_threshold = 0.9
        self.gaussblr_filt = (5, 5)
        self.meansub_filt = 4
        self.morph_filt = (4,4)

    def norm(self, data):
        mn = data.mean()
        std = data.std()
        return((data-mn)/std)

    def rescale(self, data):
        return (data-data.min())/(data.max()-data.min())

    def quantfilt(self, data, thr=None):
        if thr is None: thr = self.quantfilt_threshold
        minimum = data.min()
        filt = np.quantile(data,thr,axis=1,keepdims=True)
        out = np.where(data<filt,minimum,data)
        return out

    # gaussian filtering
    def gaussblr(self, data, filt=None):
        if filt is None: filt = self.gaussblr_filt
        out = gaussian_filter(data, filt)
        return self.rescale(out)

    # mean filtering
    def meansub(self, data):
        mn = np.mean(data)
        out = data - mn
        out = np.absolute(data - mn)
        return self.rescale(out)

    # morphological filtering
    def morph(self, data):
        se1 = footprint_rectangle(
            (self.morph_filt[0], self.morph_filt[1])
            )
        se2 = footprint_rectangle(
            (self.morph_filt[0], self.morph_filt[1])
            )
        mask = closing(data, se1)
        mask = opening(mask, se2)
        return self.rescale(mask)

    def __call__(self, x):
        x = self.quantfilt(x)
        x = self.gaussblr(x)
        x = self.meansub(x)
        x = self.morph(x)
        x = self.meansub(x)
        return x

class KourocheSpectral():
    def __init__(self):
        self.threshold = 1.5
        self.gate_smooth = 2.0
        self.gate_factor = 0.9

    def __call__(self, x):
        mean, std = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True)
        x_gate = x > (mean + self.threshold * std)
        x_gate = x_gate.astype(np.float32)
        x_gate = gaussian_filter(x_gate, self.gate_smooth)
        x_gate = (x_gate - x_gate.min()) / (x_gate.max() - x_gate.min())
        x = x * (x_gate * self.gate_factor + (1 - self.gate_factor))
        return x