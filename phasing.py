import sys
import numpy as np
from scipy import ndimage

class ModePhaser():
    def __init__(self, intens, num_supp=None, maxfrac=None):
        self.fobs = np.empty_like(intens)
        self.fobs[intens>=0] = np.sqrt(intens[intens>=0])
        self.fobs[intens<0] = -1
        self.rel_qpix = (self.fobs >= 0)

        if maxfrac is None and num_supp is None:
            raise ValueError('Need wither num_supp or maxfrac for shrinkwrap')
        elif maxfrac is None:
            self.num_supp = num_supp
            self.maxfrac = None
        elif num_supp is None:
            self.maxfrac = maxfrac
            self.num_supp = None
        else:
            raise ValueError('Cannot use both num_supp and maxfrac')

    def phase(self, algorithms):
        self.current = np.random.random(self.fobs.shape)
        for i, algo in enumerate(algorithms):
            self.run_iteration(algo)
            sys.stderr.write('\rIteration %d/%d: %s' % (i+1, len(algorithms), algo))
        sys.stderr.write('\n')

        # Shift support center to center of array
        #x, y = np.indices(dens.shape)
        #com = x*self.support/self.num_supp, y*self.support/self.num_supp
        #np.roll(self.current, cen - com, axis=(0,1))
        #np.roll(self.support, cen - com, axis=(0,1))
    
    def run_iteration(self, algo='DM'):
        if algo == 'ER':
            self.current = self.proj_fourier(self.proj_direct(self.current))
        elif algo == 'DM':
            p1 = self.proj_fourier(self.current)
            p2 = self.proj_direct(2 * p1 - self.current)
            self.current += p2 - p1
        else:
            raise ValueError('Unknown algorithm name: %s' % algo)

    def proj_fourier(self, dens):
        fdens = np.fft.fftshift(np.fft.fftn(dens))
        sel = self.rel_qpix & (np.abs(fdens) > 0)
        fdens[sel] *= self.fobs[sel] / np.abs(fdens[sel])
        return np.real(np.fft.ifftn(np.fft.ifftshift(fdens)))

    def proj_direct(self, dens):
        out_dens = np.copy(dens)

        # Shrinkwrap / Volume constraint
        smdens = ndimage.gaussian_filter(dens, 3)
        if self.num_supp is not None:
            thresh = np.sort(smdens.ravel())[-self.num_supp]
        else:
            thresh = smdens.max() * self.maxfrac
        self.support = smdens > thresh
        out_dens[~self.support] = 0

        # Positivity
        out_dens[out_dens < 0] = 0

        return out_dens
