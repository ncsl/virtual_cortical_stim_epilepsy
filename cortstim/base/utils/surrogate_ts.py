import random

import mne
import numpy as np

from cortstim.base.config.masterconfig import Config
from cortstim.base.utils.log_error import initialize_logger


class SurrogateTS():
    def __init__(self, config=None):
        self.config = config or Config
        self.logger = initialize_logger(Config.__name__)

    def digital_filtered_surrogate(self, ts, samplerate, n_iterations):
        """
        0. Estimate power spectra using overlapping FT (e.g. Welch’s method). Demean the data
        1. Take square root of power spectrum estimate and it’s inverse FT = response function
        2. Make random permutaiton of data (shuffled surrogate) and demean it.
        3. Digital filter shuffled surrogate with response function by convolution.
        4. Rescale filtered surrogate to original data distribution (same as AAFT procedure)

        :param ts:
        :param n_iterations:
        :return:
        """
        # 1. compute power spectra with overlapping FT - e.g. Welch's method
        fmin = 0.5
        fmax = samplerate // 2
        power, freqs = mne.time_frequency.psd_array_welch(ts, sfreq=samplerate,
                                                          fmin=fmin,
                                                          fmax=fmax)

        # 2. compute response function
        resp_func = np.sqrt(power)

        # 3. make random permutation of the data (e.g. whitenoiseshuffled_surrogate)
        whitenoise_surrogate = self.whitenoiseshuffled_surrogate(ts)
        whitenoise_surrogate = whitenoise_surrogate - \
            np.mean(whitenoise_surrogate)

        # 4. convolve whitenoise surrogate with response function
        convolved_ts = np.convolve(resp_func, whitenoise_surrogate)

        #  4. Rescale data to original data distibution
        ranks = ts.argsort(axis=1).argsort(axis=1)
        rescaled_data = np.zeros(ts.shape)
        for i in range(ts.shape[0]):
            rescaled_data[i, :] = convolved_ts[i, ranks[i, :]]

        #  Phase randomize rescaled data
        phase_randomized_data = \
            self.phaseshuffled_surrogate(rescaled_data)

        #  Rescale back to amplitude distribution of original data
        sorted_original = ts.copy()
        sorted_original.sort(axis=1)

        ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

        for i in range(ts.shape[0]):
            rescaled_data[i, :] = sorted_original[i, ranks[i, :]]
        return rescaled_data

    def aaft_surrogate(self, ts):
        """
        Return surrogates using the amplitude adjusted Fourier transform
        method.

        1. Simulate first f
        −1
        , by reordering white noise data to have same rank structure as x
        2. make randomized copy using a FT surrogate, y
        FT
        3. and then transform it back to f . -> reorder x to have same rank strcuture as y
        FT
        Reference: [Schreiber2000]_

        :type ts: 2D array [index, time]
        :arg ts: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        #  Create sorted Gaussian reference series
        gaussian = np.random.randn(ts.shape[0], ts.shape[1])
        gaussian.sort(axis=1)

        #  Rescale data to Gaussian distribution
        ranks = ts.argsort(axis=1).argsort(axis=1)
        rescaled_data = np.zeros(ts.shape)

        for i in range(ts.shape[0]):
            rescaled_data[i, :] = gaussian[i, ranks[i, :]]

        #  Phase randomize rescaled data
        phase_randomized_data = \
            self.phaseshuffled_surrogate(rescaled_data)

        #  Rescale back to amplitude distribution of original data
        sorted_original = ts.copy()
        sorted_original.sort(axis=1)

        ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

        for i in range(ts.shape[0]):
            rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

        return rescaled_data

    def iaaft_surrogate(self, ts, n_iterations, output="true_amplitudes"):
        """
        Return surrogates using the iteratively refined amplitude adjusted
        Fourier transform method.

        A set of AAFT surrogates (:meth:`AAFT_surrogates`) is iteratively
        refined to produce a closer match of both amplitude distribution and
        power spectrum of surrogate and original data.

        Reference: [Schreiber2000]_

        :type ts: 2D array [index, time]
        :arg ts: The original time series.
        :arg int n_iterations: Number of iterations / refinement steps
        :arg str output: Type of surrogate to return. "true_amplitudes":
            surrogates with correct amplitude distribution, "true_spectrum":
            surrogates with correct power spectrum, "both": return both outputs
            of the algorithm.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        #  Get size of dimensions
        n_time = ts.shape[1]

        #  Get Fourier transform of original data with caching
        if self._fft_cached:
            fourier_transform = self._ts_fft
        else:
            fourier_transform = np.fft.rfft(ts, axis=1)
            self._ts_fft = fourier_transform
            self._fft_cached = True

        #  Get Fourier amplitudes
        original_fourier_amps = np.abs(fourier_transform)

        #  Get sorted copy of original data
        sorted_original = ts.copy()
        sorted_original.sort(axis=1)

        #  Get starting point / initial conditions for R surrogates
        # (see [Schreiber2000]_)
        R = self.aaft_surrogate(ts)

        #  Start iteration
        for i in range(n_iterations):
            #  Get Fourier phases of R surrogate
            r_fft = np.fft.rfft(R, axis=1)
            r_phases = r_fft / np.abs(r_fft)

            #  Transform back, replacing the actual amplitudes by the desired
            #  ones, but keeping the phases exp(iψ(i)
            s = np.fft.irfft(original_fourier_amps * r_phases, n=n_time,
                             axis=1)

            #  Rescale to desired amplitude distribution
            ranks = s.argsort(axis=1).argsort(axis=1)

            for j in range(ts.shape[0]):
                R[j, :] = sorted_original[j, ranks[j, :]]

        if output == "true_amplitudes":
            return R
        elif output == "true_spectrum":
            return s
        elif output == "both":
            return (R, s)
        else:
            return (R, s)

    def phaseshuffled_surrogate(self, ts):
        """
        Return Fourier surrogates.

        Generate surrogates by Fourier transforming the :attr:`ts`
        time series (assumed to be real valued), randomizing the phases and
        then applying an inverse Fourier transform. Correlated noise surrogates
        share their power spectrum and autocorrelation function with the
        ts time series.

        The Fast Fourier transforms of all time series are cached to facilitate
        a faster generation of several surrogates for each time series. Hence,
        :meth:`clear_cache` has to be called before generating surrogates from
        a different set of time series!

        .. note::
           The amplitudes are not adjusted here, i.e., the
           individual amplitude distributions are not conserved!

        **Examples:**

        The power spectrum is conserved up to small numerical deviations:

        >>> ts = Surrogates.SmallTestData().ts
        >>> surrogates = Surrogates.\
                SmallTestData().correlated_noise_surrogates(ts)
        >>> all(r(np.abs(np.fft.fft(ts,         axis=1))[0,1:10]) == \
                r(np.abs(np.fft.fft(surrogates, axis=1))[0,1:10]))
        True

        However, the time series amplitude distributions differ:

        >>> all(np.histogram(ts[0,:])[0] == np.histogram(surrogates[0,:])[0])
        False

        :type ts: 2D array [index, time]
        :arg ts: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        self.logger.debug("Generating correlated noise surrogates...")

        #  Calculate FFT of ts time series
        #  The FFT of the ts data has to be calculated only once,
        #  so it is stored in self._ts_fft.
        if self._fft_cached:
            surrogates = self._ts_fft
        else:
            surrogates = np.fft.rfft(ts, axis=1)
            self._ts_fft = surrogates
            self._fft_cached = True

        #  Get shapes
        (N, n_time) = ts.shape
        len_phase = surrogates.shape[1]

        #  Generate random phases uniformly distributed in the
        #  interval [0, 2*Pi]
        phases = random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

        #  Add random phases uniformly distributed in the interval [0, 2*Pi]
        surrogates *= np.exp(1j * phases)

        #  Calculate IFFT and take the real part, the remaining imaginary part
        #  is due to numerical errors.
        return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                         axis=1)))

    def whitenoiseshuffled_surrogate(self, ts):
        """
        Return a shuffled copy of a time series array.

        Each time series is shuffled individually. The surrogates correspond to
        realizations of white noise consistent with the :attr:`ts`
        time series' amplitude distribution.

        **Example** (Distributions of white noise surrogates should the same as
        for the original data):

        >>> ts = Surrogates.SmallTestData().ts
        >>> surrogates = Surrogates.\
                SmallTestData().white_noise_surrogates(ts)
        >>> np.histogram(ts[0,:])[0]
        array([21, 12,  9, 15, 34, 35, 18, 12, 16, 28])
        >>> np.histogram(surrogates[0,:])[0]
        array([21, 12,  9, 15, 34, 35, 18, 12, 16, 28])

        :type ts: 2D array [index, time]
        :arg ts: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        self.logger.debug(
            "Generating white noise surrogates by random shuffling...")

        #  Generate reference to shuffle function
        shuffle = random.shuffle

        surrogates = ts.copy()

        for i in range(surrogates.shape[0]):
            shuffle(surrogates[i, :])

        return surrogates
