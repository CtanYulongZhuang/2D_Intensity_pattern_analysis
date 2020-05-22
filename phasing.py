import numpy as np
from scipy import ndimage

class ModePhaser():
    def __init__(self, intens):
        self.fobs = np.empty_like(intens)
        self.fobs[intens>0] = np.sqrt(intens[intens>0])
        self.fobs[intens<0] = -1
        self.rel_qpix = (self.fobs >= 0)

    def phase(self, algorithms):
        self.current = np.random.random(intens.shape)
        for algo in algorithms:
            self.run_iteration(algo)
    
    def run_iteration(self, algo='DM'):
        if algo == 'ER':
            self.current = self.proj_direct(self.proj_fourier(self.current))
        elif algo == 'DM':
            p1 = self.proj_fourier(self.current)
            p2 = self.proj_direct(2 * p1 - self.current)
            self.current += p2 - p1
        else:
            raise ValueError('Unknown algorithm name: %s' % algo)

    def proj_fourier(self, dens):
        fdens = np.fft.fftshift(np.fft.fftn(dens))
        fdens[self.rel_qpix] *= self.fobs[self.rel_qpix] / np.abs(fdens[self.rel_qpix])
        return np.real(np.fft.fftshift(np.fft.ifftn(fdens)))

    def proj_direct(self, dens):
        out_dens = np.copy(dens)
        smdens = ndimage.gaussian_filter(dens, 3)
        self.support = smdens > 0.2 * smdens.max()
        out_dens[~self.support] = 0
        return out_dens
