"""
A SineGenerator module.

A SineGenerator object can be initiated to return discrete samples for a sine wave.

The sampling rate, frequency, amplitude and instantaneous phase are adjustable.

Implemented with duck typing approach (no type checking, only necessary value checking). Additionally, prototypes are used extensively to ensure simple usage.

TODO:
    - Implement dynamic data type

Author:
    Name: Johannes Engell Kamber
    Last edited: 2016-09-30
    License: Published under GNU GPLv3
"""

import math
import warnings
import numpy as np

class SineGenerator:
    """
    Creates a SineGenerator object which is used to generate descrete sine waves
    """

    def __init__(self, fs=None, f=None, amp=None, phase=None):
        """
        Initiate the SineGenetor object with a set of attributes.

        Attributes:
            fs (optional [int]): The sampling frequency. Must be larger than 0. (default 48000)
            f (optional [number]): Frequency. (default 500)
            amp (optional [number]): Amplitude. (default 1)
            phase (optional [number]): Initial phase shift in radians(default 0)

        Examples:
            >>> sg = SineGenerator()
            >>> sg = SineGenerator(44100, 200, 0.2, -math.pi)
            >>> sg = SineGenerator(amp = 0.2)
        """
        if fs is None:
            fs = 48000
        if f is None:
            f = 500
        if amp is None:
            amp = 1
        if phase is None:
            phase = 0
        self._fs = fs
        self._f = f
        self._amp = amp
        self._phase = phase

        self.fs = fs
        self.f = f
        self.amp = amp
        self.phase = phase

    def get_samples(self, N=None, T=None, dtype=None):
        """
        Return a number of discrete samples as a NumPy array.

        The method updates the current phase of the SineGenerator, so that
        subsequent calls to    get_samples() are appendable.
        If the frequency 0 is set, the returned values will represent a DC
        signal of the set amplitude.

        Arguments:
            N (optional [number]): Number of discrete samples. Must be larger than 0 (default fs). N is rounded to nearest integer.
            T (optional [number]): Total time of returned samples. Must be larger than 0. (default 1)
            dtype (optional [numpy.dtype]): The data type of returned samples (default numpy.float32

            If both N and T is supplied, T is ignored.

        Returns:
            numpy.ndarray: Array of data in the given data type

        Examples:
            >>> sg = SineGenerator()
            >>> x = sg.get_samples(2) # Get 2 discrete samples. Equivalent to sg.get_samles(N=2)
            >>> x
            [0., 0.01]
            >>> y = sg.get_samles(T=3) # Get 3 seconds of discrete data
            >>> y
            [0.2, 0.03, ...] # Can be appended to x, leading to a continuous sine wave
            >>> x.dtype
            dtype('float32')
            >>> type(x)
            <type 'numpy.ndarray'>

        """
        if N is None and T is None:
            N = self.fs
        if N is None and T is not None:
            N = T * self.fs
        if N is not None and T is not None:
            if T * self.fs != N:
                raise ValueError('Specified N and T lead to different sample sizes')
        if dtype is None:
            dtype = np.float32
        if N <= 0:
            raise ValueError('The number of samples cannot be 0')

        t = np.arange(0, N) / self.fs
        sig = self.amp * np.sin(2*np.pi * self.f * t + self.phase)
        self.phase = self.phase + 2*np.pi*N/self.fs*self.f
        return sig

    @property
    def fs(self):
        """
        Sampling frequency property (read/write).

        Raises:
            ValueError: If the supplied sampling frequency is not larger than 0

        Warnings:
            Issues a warning if the Nyquist frequency is smaller than the set frequency
        """
        return self._fs

    @fs.setter
    def fs(self, fs):
        if fs < 0:
            raise ValueError('The sampling frequency must be larger than 0')
        if fs < self.f*2:
            warnings.warn('The Nyquist frequency is smaller than the frequency f')
        self._fs = fs

    @property
    def f(self):
        """
        Frequency property (read/write).

        Raises:
            ValueError: If the supplied frequency is smaller than 0

        Warnings:
            Issues a wasrning if the frequency is larger than the Nyquist frequency
        """
        return self._f

    @f.setter
    def f(self, f):
        if f < 0:
            raise ValueError('The frequency must be larger than or equal to 0')
        if f > self.fs/2:
            warnings.warn('The frequency is larger than half the sampling frequency fs')
        self._f = f

    @property
    def amp(self):
        """Amplitude property (read/write)."""
        return self._amp

    @amp.setter
    def amp(self, amp):
        self._amp = amp

    @property
    def phase(self):
        """
        Phase property (read/write).

        Represents the current phase of the sine generator in radians.
        Any number is accepted and adjusted to be in the interval [0.; math.pi*2[
        """
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase % (np.pi*2)

print(__name__)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sig = np.array([])
    sg1 = SineGenerator(fs=100, f=12.5)
    sg2 = SineGenerator(fs=100, f=1)


    sig = sg1.get_samples(T=10) + sg2.get_samples(T=10)
    
    plt.plot(sig)
    plt.show()