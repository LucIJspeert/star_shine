"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains functions that evaluate certain configuration attributes that depend on data properties.

Code written by: Luc IJspeert
"""
import numpy as np
import numba as nb

from star_shine.config.helpers import get_config


# load configuration
config = get_config()


def signal_to_noise_threshold(time):
    """Determine the signal-to-noise threshold for accepting frequencies based on the number of points

    Based on Baran & Koen 2021, eq 6. (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)

    snr_thr = 1.201 * np.sqrt(1.05 * np.log(len(time)) + 7.184)
    Plus 0.25 in case of gaps > 27 days

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.

    Returns
    -------
    float
        signal-to-noise threshold for this data set.
    """
    # if user defined (unequal to -1), return the configured number
    if config.snr_thr != -1:
        return config.snr_thr

    # equation 6 from Baran & Koen 2021
    snr_thr = 1.201 * np.sqrt(1.05 * np.log(len(time)) + 7.184)

    # increase threshold by 0.25 if gaps longer than 27 days
    if np.any((time[1:] - time[:-1]) > 27):
        snr_thr += 0.25

    # round to two decimals
    snr_thr = np.round(snr_thr, 2)

    return snr_thr


def frequency_resolution(time, factor=1.5):
    """Calculate the frequency resolution of a time series

    Equation: factor / T, where T is the total time base of observations.
    Recommended factor: 1.5. Common choices are 1, 1.5 (conservative), 2.5 (very conservative).

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    factor: float, optional
        Number that multiplies the resolution (1/T).

    Returns
    -------
    float
        Frequency resolution of the time series
    """

    f_res = factor / np.ptp(time)

    return f_res


def frequency_lower_threshold(time, factor=0.01):
    """Calculate the frequency resolution of a time series

    Equation: factor / T, where T is the total time base of observations.
    Recommended factor: 1/100.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    factor: float, optional
        Number that multiplies the resolution (1/T).

    Returns
    -------
    float
        Frequency resolution of the time series
    """

    f_min = factor / np.ptp(time)

    return f_min


# @nb.njit(cache=True)
def nyquist_sum_koen_2006(n, time, delta_t_min):
    """"""
    factor = n * np.pi / delta_t_min

    # evaluate equation 5 from Koen 2006 at nu = 2pi*n/delta_t_min
    ss = 0
    for i in range(0, len(time) - 1):
        for j in range(i + 1, len(time)):
            ss += np.sin(factor * (time[j] - time[i]))

    return ss


def nyquist_frequency(time):
    """Determines the maximum frequency for extraction and periodograms.

    The Nyquist frequency is calculated using the configured built-in function.
    The rigorous method implements equation 5 from Koen 2006.
    https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1390K/abstract

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.

    Returns
    -------
    float
        Nyquist frequency of the time series
    """
    # smallest time interval
    delta_t_min = np.min(time[1:] - time[:-1])

    # calculate the Nyquist frequency with the specified approach
    if config.nyquist == 'rigorous':
        # iterate n until sum returns zero
        ss_nu = 0
        n = 0
        while ss_nu == 0:
            n += 1
            ss_nu = nyquist_sum_koen_2006(n, time, delta_t_min)

        f_nyquist = n / (2 * delta_t_min)
    else:
        # method == 'simple'
        f_nyquist = 1 / (2 * delta_t_min)

    return f_nyquist
