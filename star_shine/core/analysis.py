"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains algorithms for data analysis.

Code written by: Luc IJspeert
"""
import numpy as np

from star_shine.core import time_series as tms, model as mdl, periodogram as pdg
from star_shine.core import fitting as fit, goodness_of_fit as gof, frequency_sets as frs
from star_shine.core import utility as ut
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


def extract_single(time, flux, f0=-1, fn=-1, select='a'):
    """Extract a single sinusoid from a time series.

    The extracted frequency is based on the highest amplitude or signal-to-noise in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float, optional
        Lowest allowed frequency for extraction.
        If left -1, default is f0 = 1/(100*T)
    fn: float, optional
        Highest allowed frequency for extraction.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    select: str, optional
        Select the next frequency based on amplitude 'a' or signal-to-noise 'sn'

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid

    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # full LS periodogram
    freqs, ampls = pdg.scargle_parallel(time, flux, f0=f0, fn=fn, df=df)

    # selection step based on flux to noise (refine step keeps using ampl)
    if select == 'sn':
        noise_spectrum = pdg.scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0)
        ampls = ampls / noise_spectrum

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df/100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_local(time, flux, f0, fn):
    """Extract a single sinusoid from a time series at a predefined frequency interval.

    The extracted frequency is based on the highest amplitude in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Lowest allowed frequency for extraction.
    fn: float
        Highest allowed frequency for extraction.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid

    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # full LS periodogram (accurate version)
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # cut off the ends of the frequency range if they are rising
    i_f_min_edges = ut.uphill_local_max(freqs, -ampls, freqs[np.array([0, -1])])
    freqs = freqs[i_f_min_edges[0]:i_f_min_edges[1] + 1]
    ampls = ampls[i_f_min_edges[0]:i_f_min_edges[1] + 1]

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_approx(time, flux, f_approx):
    """Extract a single sinusoid from a time series at an approximate location.

    Follows the periodogram upwards to the nearest peak. The periodogram is oversampled for a more precise result.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_approx: float
        Approximate location of the frequency of maximum amplitude.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # LS periodogram around the approximate location (2.5 times freq res)
    f0 = max(f_approx - 25 * df, df / 10)
    fn = f_approx + 25 * df
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # get the index of the frequency of the maximum amplitude
    i_f_max = ut.uphill_local_max(freqs, ampls, np.array([f_approx]))[0]

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def refine_subset(ts_model, close_f, logger=None):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other,
    taking into account (and not changing the frequencies of) harmonics if present.

    Intended as a sub-loop within another extraction routine (extract_sinusoids), can work standalone too.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    close_f: numpy.ndarray[Any, dtype[int]]
        Indices of the subset of frequencies to be refined
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    See Also
    --------
    extract_sinusoids
    """
    # get all the harmonics (regardless of base)
    i_harmonics = np.arange(len(ts_model.sinusoid.harmonics))[ts_model.sinusoid.harmonics]

    # determine initial bic
    bic_prev = ts_model.bic()

    # stop the loop when the BIC increases
    condition_1 = True
    while condition_1:
        # remember the current sinusoids
        f_c, a_c, ph_c = ts_model.sinusoid.get_sinusoid_parameters(exclude=False)

        # remove each frequency one at a time to then re-extract them
        for j in close_f:
            # exclude the sinusoid
            ts_model.exclude_sinusoids(j)
            # update the linear model for good measure
            ts_model.update_linear_model()

            # improve sinusoid j by re-extracting its parameters
            f_j = ts_model.sinusoid.f_n[j]
            if j in i_harmonics:
                # if f is a harmonic, don't shift the frequency
                a_j, ph_j = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_j)
            else:
                f_j, a_j, ph_j = extract_approx(ts_model.time, ts_model.residual(), f_j)

            # update the model
            ts_model.update_sinusoids(f_j, a_j, ph_j, j)

        # as a last model-refining step, redetermine the constant and slope
        ts_model.update_linear_model()

        # calculate BIC before moving to the next iteration
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # stop the loop when the BIC increases
        condition_1 = np.round(d_bic, 2) > 0

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            # update the sinusoids back to their original values
            ts_model.update_sinusoids(f_c, a_c, ph_c, close_f)
            ts_model.update_linear_model()

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Refined: {len(close_f)}")

    return ts_model

def replace_subset(ts_model, close_f, logger=None):
    """Attempt the replacement of frequencies within the Rayleigh criterion of each other by a single one,
    taking into account (and not changing the frequencies of) harmonics if present.

    Intended as a sub-loop within another extraction routine (extract_sinusoids), can work standalone too.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    close_f: numpy.ndarray[Any, dtype[int]]
        Indices of the subset of frequencies to be (subdivided and) replaced.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    See Also
    --------
    extract_sinusoids
    """
    # standard frequency resolution (not the user defined one)
    freq_res = 1 / ts_model.t_tot
    # get all the harmonics (regardless of base)
    i_harmonics = np.arange(len(ts_model.sinusoid.harmonics))[ts_model.sinusoid.harmonics]

    # make all combinations of consecutive frequencies in close_f (longer sets first)
    close_f_sets = ut.consecutive_subsets(close_f)

    # determine initial bic
    bic_prev = ts_model.bic()

    # loop over all subsets:
    removed = []
    for set_i in close_f_sets:
        # if set_i contains removed sinusoids, skip (order of sets matters)
        if np.any([i in removed for i in set_i]):
            continue

        # exclude the next set of sinusoids
        ts_model.exclude_sinusoids(set_i)
        # update the linear model for good measure
        ts_model.update_linear_model()

        # check for harmonics
        harm_i = [h for h in set_i if h in i_harmonics]

        # remove all frequencies in the set and re-extract one
        f_c = ts_model.sinusoid.f_n  # current frequencies
        if len(harm_i) > 0:
            # if f is a harmonic, don't shift the frequency
            f_i = f_c[harm_i]  # can be more than one harmonic
            a_i, ph_i = pdg.scargle_ampl_phase(ts_model.time, ts_model.residual(), f_i)
        else:
            f0 = min(f_c[set_i]) - freq_res
            fn = max(f_c[set_i]) + freq_res
            f_i, a_i, ph_i = extract_local(ts_model.time, ts_model.residual(), f0=f0, fn=fn)

        # add sinusoid to the model
        ts_model.add_sinusoids(f_i, a_i, ph_i)
        # as a last model-refining step, redetermine the constant and slope
        ts_model.update_linear_model()

        # calculate BIC before moving to the next iteration
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # acceptance condition for replacement
        condition_1 = np.round(d_bic, 2) > 0

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the changes
            bic_prev = bic
            removed.extend(set_i)
        else:
            # remove the added sinusoid(s)
            ts_model.remove_sinusoids(np.arange(len(f_c), len(f_c) + len(np.atleast_1d(f_i))))

            # include the excluded sinusoids
            ts_model.include_sinusoids(set_i)
            ts_model.update_linear_model()

    ts_model.remove_excluded()

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Replaced: {len(removed)}")

    return ts_model


def extract_sinusoids(ts_model, bic_thr=2, snr_thr=0, stop_crit='bic', select='hybrid', n_extract=0,
                      fit_each_step=False, replace_each_step=True, logger=None):
    """Extract all the frequencies from a periodic flux.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    bic_thr: float, optional
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    snr_thr: float, optional
        Threshold for signal-to-noise ratio for a signal to be considered significant.
    stop_crit: str, optional
        Use the BIC as stopping criterion or the SNR, choose from 'bic', or 'snr'
    select: str, optional
        Select the next frequency based on amplitude ('a'),
        signal-to-noise ('sn'), or hybrid ('hybrid') (first a then sn).
    n_extract: int, optional
        Maximum number of frequencies to extract. The stop criterion is still leading. Zero means as many as possible.
    fit_each_step: bool
        If set to True, a non-linear least-squares fit of all extracted sinusoids in groups is performed at each
        iteration. While this increases the quality of the extracted signals, it drastically slows down the code.
        Generally gives a better quality of the extraction than only doing this all the way at the end.
    replace_each_step: bool
        If set to True, close frequecies are attempted to be replaced by a single sinusoid at each iteration.
        May increase the quality of the extraction more than only doing this all the way at the end.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    Notes
    -----
    Spits out frequencies and amplitudes in the same units as the input,
    and phases that are measured with respect to the first time point.
    Also determines the flux average, so this does not have to be subtracted
    before input into this function.
    Note: does not perform a non-linear least-squares fit at the end,
    which is highly recommended! (In fact, no fitting is done at all).

    The function optionally takes a pre-existing frequency list to append
    additional frequencies to. Set these to np.array([]) to start from scratch.

    i_chunks is a 2D array with start and end indices of each (half) sector.
    This is used to model a piecewise-linear trend in the data.
    If you have no sectors like the TESS mission does, set
    i_chunks = np.array([[0, len(time)]])

    Exclusively uses the Lomb-Scargle periodogram (and an iterative parameter
    improvement scheme) to extract the frequencies.
    Uses a delta BIC > bic_thr stopping criterion.

    [Author's note] Although it is my belief that doing a non-linear
    multi-sinusoid fit at each iteration of the prewhitening is the
    ideal approach, it is also a very (very!) time-consuming one and this
    algorithm aims to be fast while approaching the optimal solution.

    [Another author's note] I added an option to do the non-linear multi-
    sinusoid fit at each iteration.
    """
    if n_extract == 0:
        n_extract = 10**6  # 'a lot'

    # initial number of sinusoids
    n_sin_init = ts_model.sinusoid.n_sin

    # set up selection process
    if select == 'hybrid':
        switch = True  # when we would normally end, we switch strategy
        select = 'a'  # start with amplitude extraction
    else:
        switch = False

    # determine the initial bic
    bic_prev = ts_model.bic()  # initialise current BIC to the mean (and slope) subtracted flux
    bic_init = bic_prev

    # log a message
    if logger is not None:
        logger.extra(f"N_f= {n_sin_init}, BIC= {bic_init:1.2f}")

    # stop the loop when the BIC decreases by less than 2 (or increases)
    condition_1 = True
    condition_2 = True
    while (condition_1 | switch) & condition_2:
        # switch selection method when extraction would normally stop
        if switch and not condition_1:
            select = 'sn'
            switch = False

        # remember the current sinusoids
        f_c, a_c, ph_c = ts_model.sinusoid.get_sinusoid_parameters(exclude=False)

        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(ts_model.time, ts_model.residual(), f0=ts_model.pd_f0, fn=ts_model.pd_fn,
                                        select=select)
        ts_model.add_sinusoids(f_i, a_i, ph_i)

        # imporve frequencies with some strategy
        if fit_each_step:
            # fit all frequencies for best improvement
            out = fit.fit_multi_sinusoid_per_group(ts_model.time, ts_model.flux, *ts_model.get_parameters(),
                                                   ts_model.i_chunks, logger=logger)

            ts_model.set_linear_model(out[0], out[1])
            ts_model.set_sinusoids(out[2], out[3], out[4])
        else:
            # select only close frequencies for iteration
            close_f = frs.f_within_rayleigh(ts_model.sinusoid.n_sin - 1, ts_model.sinusoid.f_n, ts_model.f_resolution)

            if len(close_f) > 1:
                # iterate over (re-extract) close frequencies (around f_i) a number of times to improve them
                ts_model = refine_subset(ts_model, close_f, logger=logger)
            else:
                # only update the linear pars
                ts_model.update_linear_model()

        # possibly replace close frequencies
        if replace_each_step:
            close_f = frs.f_within_rayleigh(ts_model.sinusoid.n_sin - 1, ts_model.sinusoid.f_n, ts_model.f_resolution)
            if len(close_f) > 1:
                ts_model = replace_subset(ts_model, close_f, logger=logger)

        # calculate BIC
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # acceptance condition
        if stop_crit == 'snr':
            # calculate SNR in a 1 c/d window around the extracted frequency
            noise = pdg.scargle_noise_at_freq(np.array([f_i]), ts_model.time, ts_model.residual(), window_width=1.0)
            snr = a_i / noise
            # stop the loop if snr threshold not met
            condition_1 = snr > snr_thr
        else:
            # stop the loop when the BIC decreases by less than bic_thr (or increases)
            condition_1 = np.round(d_bic, 2) > bic_thr

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            ts_model.set_sinusoids(f_c, a_c, ph_c)
            ts_model.update_linear_model()

        # stop the loop if n_sin reaches limit
        condition_2 = ts_model.sinusoid.n_sin - n_sin_init < n_extract

        if logger is not None and condition_1:
            logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - Extracted: f= {f_i:1.6f}, a= {a_i:1.6f}")

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Extraction ended "
                     f"(delta BIC= {bic_init - bic_prev:1.2f}).")

    return ts_model


def extract_harmonics(ts_model, bic_thr=2, logger=None):
    """Tries to extract more harmonics from the flux

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    bic_thr: float
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    See Also
    --------
    fix_harmonic_frequency

    Notes
    -----
    Looks for missing harmonics and checks whether adding them decreases the BIC sufficiently (by more than 2).
    Assumes the harmonics are already fixed multiples of 1/p_orb as can be achieved with fix_harmonic_frequency.
    """
    # make lists of not-present possible harmonics paired with their base frequency
    f_base = []
    h_candidates_n = []
    for i_base in np.unique(ts_model.sinusoid.h_base[ts_model.sinusoid.harmonics]):
        # the range of harmonic multipliers below the Nyquist frequency
        harmonics_i = np.arange(1, ts_model.pd_fn / ts_model.sinusoid.f_n[i_base], dtype=int)

        # harmonic_n minus one is the position for existing harmonics
        harmonics_i = np.delete(harmonics_i, ts_model.sinusoid.harmonic_n[ts_model.sinusoid.h_base == i_base] - 1)
        h_candidates_n.extend(harmonics_i)
        f_base.extend([ts_model.sinusoid.f_n[i_base] for _ in range(len(harmonics_i))])

    # determine the initial bic
    bic_prev = ts_model.bic()  # initialise current BIC to the mean (and slope) subtracted flux
    bic_init = bic_prev

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_init:1.2f}")

    # loop over candidates and try to extract (BIC decreases by 2 or more)
    for i in range(len(h_candidates_n)):
        f_i = h_candidates_n[i] * f_base[i]
        a_i, ph_i = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_i)
        ts_model.add_sinusoids(f_i, a_i, ph_i)

        # redetermine the constant and slope
        ts_model.update_linear_model()

        # determine new BIC and whether it improved
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # stop the loop when the BIC decreases by less than bic_thr (or increases)
        condition_1 = np.round(d_bic, 2) > bic_thr

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            # h_c is rejected, revert to previous model
            ts_model.remove_sinusoids(ts_model.sinusoid.n_sin - 1)
            ts_model.update_linear_model()

        if logger is not None and condition_1:
            logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - Extracted: "
                         f"f_base= {f_base[i]:1.2f}, h= {h_candidates_n[i]}, f= {f_i:1.6f}, a= {a_i:1.6f}")

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Extraction ended "
                     f"(delta BIC= {bic_init - bic_prev:1.2f})")

    return ts_model


def fix_harmonic_frequency(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Fixes the frequency of harmonics to the theoretical value, then
    re-determines the amplitudes and phases.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of the same number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of the same number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of the same number of sine waves
    """
    # extract the harmonics using the period and determine some numbers
    freq_res = 1.5 / np.ptp(time)
    harmonics, harmonic_n = frs.find_harmonics_tolerance(f_n, p_orb, f_tol=freq_res / 2)
    if len(harmonics) == 0:
        raise ValueError('No harmonic frequencies found')

    n_chunks = len(i_chunks)
    n_sin = len(f_n)
    n_harm_init = len(harmonics)

    # indices of harmonic candidates to remove
    remove_harm_c = np.zeros(0, dtype=np.int_)
    f_new, a_new, ph_new = np.zeros((3, 0))

    # determine initial bic
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - mdl.linear_curve(time, const, slope, i_chunks)
    n_param = ut.n_parameters(n_chunks, n_sin, n_harm_init)
    bic_init = gof.calc_bic(resid, n_param)

    # go through the harmonics by harmonic number and re-extract them (removing all duplicate n's in the process)
    for n in np.unique(harmonic_n):
        remove = np.arange(len(f_n))[harmonics][harmonic_n == n]
        # make a model of the removed sinusoids and subtract it from the full sinusoid residual
        model_sinusoid_r = mdl.sum_sines(time, f_n[remove], a_n[remove], ph_n[remove])
        cur_resid += model_sinusoid_r
        const, slope = mdl.linear_pars(time, resid, i_chunks)  # redetermine const and slope
        resid = cur_resid - mdl.linear_curve(time, const, slope, i_chunks)

        # calculate the new harmonic
        f_i = n / p_orb  # fixed f
        a_i, ph_i = pdg.scargle_ampl_phase_single(time, resid, f_i)

        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = mdl.sum_sines(time, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        cur_resid -= model_sinusoid_n

        # add to freq list and removal list
        f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
        remove_harm_c = np.append(remove_harm_c, remove)
        if logger is not None:
            logger.extra(f"Harmonic number {n} re-extracted, replacing {len(remove)} candidates")

    # lastly re-determine slope and const (not needed here)
    # const, slope = mdl.linear_pars(time, cur_resid, i_chunks)
    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.append(np.delete(f_n, remove_harm_c), f_new)
    a_n = np.append(np.delete(a_n, remove_harm_c), a_new)
    ph_n = np.append(np.delete(ph_n, remove_harm_c), ph_new)

    # re-extract the non-harmonics
    n_sin = len(f_n)
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_sin), harmonics)
    n_harm = len(harmonics)
    remove_non_harm = np.zeros(0, dtype=np.int_)
    for i in non_harm:
        # make a model of the removed sinusoid and subtract it from the full sinusoid residual
        model_sinusoid_r = mdl.sum_sines(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid += model_sinusoid_r
        const, slope = mdl.linear_pars(time, cur_resid, i_chunks)  # redetermine const and slope
        resid = cur_resid - mdl.linear_curve(time, const, slope, i_chunks)

        # extract the updated frequency
        fl, fr = f_n[i] - freq_res, f_n[i] + freq_res
        f_n[i], a_n[i], ph_n[i] = extract_approx(time, resid, f_n[i])
        if (f_n[i] <= fl) | (f_n[i] >= fr):
            remove_non_harm = np.append(remove_non_harm, [i])

        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = mdl.sum_sines(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid -= model_sinusoid_n

    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.delete(f_n, non_harm[remove_non_harm])
    a_n = np.delete(a_n, non_harm[remove_non_harm])
    ph_n = np.delete(ph_n, non_harm[remove_non_harm])

    # re-establish cur_resid
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    const, slope = mdl.linear_pars(time, cur_resid, i_chunks)  # lastly re-determine slope and const

    if logger is not None:
        resid = cur_resid - mdl.linear_curve(time, const, slope, i_chunks)
        n_param = ut.n_parameters(n_chunks, n_sin, n_harm)
        bic = gof.calc_bic(resid, n_param)
        logger.extra(f"Candidate harmonics replaced: {n_harm_init} ({n_harm} left). "
                     f"N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {bic_init - bic:1.2f})")

    return const, slope, f_n, a_n, ph_n


def remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt the removal of individual frequencies

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves

    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency.
    Harmonics are taken into account.
    """
    n_chunks = len(i_chunks)
    n_sin = len(f_n)
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)

    # indices of single frequencies to remove
    remove_single = np.zeros(0, dtype=np.int_)

    # determine initial bic
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - mdl.linear_curve(time, const, slope, i_chunks)
    n_param = ut.n_parameters(n_chunks, n_sin, n_harm)
    bic_prev = gof.calc_bic(resid, n_param)
    bic_init = bic_prev
    n_prev = -1

    # while frequencies are added to the remove list, continue loop
    while len(remove_single) > n_prev:
        n_prev = len(remove_single)
        for i in range(n_sin):
            if i in remove_single:
                continue

            # make a model of the removed sinusoids and subtract it from the full sinusoid model
            model_sinusoid_r = mdl.sum_sines_st(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
            resid = cur_resid + model_sinusoid_r
            const, slope = mdl.linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= mdl.linear_curve(time, const, slope, i_chunks)

            # number of parameters and bic
            n_harm_i = n_harm - len([h for h in remove_single if h in harmonics]) - 1 * (i in harmonics)
            n_sin_i = n_sin - len(remove_single) - 1 - n_harm_i
            n_param = ut.n_parameters(n_chunks, n_sin_i, n_harm_i)
            bic = gof.calc_bic(resid, n_param)

            # if improvement, add to list of removed freqs
            if np.round(bic_prev - bic, 2) > 0:
                remove_single = np.append(remove_single, i)
                cur_resid += model_sinusoid_r
                bic_prev = bic

    # lastly re-determine slope and const
    const, slope = mdl.linear_pars(time, cur_resid, i_chunks)

    # finally, remove all the designated sinusoids from the lists
    f_n = np.delete(f_n, remove_single)
    a_n = np.delete(a_n, remove_single)
    ph_n = np.delete(ph_n, remove_single)

    if logger is not None:
        logger.extra(f"Single frequencies removed: {n_sin - len(f_n)}, "
                     f"N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f})")

    return const, slope, f_n, a_n, ph_n


def replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt the replacement of groups of frequencies by a single one

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves

    Notes
    -----
    Checks whether the BIC can be improved by replacing a group of
    frequencies by only one. Harmonics are never removed.
    """
    freq_res = 1 / np.ptp(time)  # frequency resolution
    n_chunks = len(i_chunks)
    n_sin = len(f_n)
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_sin), harmonics)
    n_harm = len(harmonics)

    # make an array of sets of frequencies (non-harmonic) to be investigated for replacement
    close_f_groups = frs.chains_within_rayleigh(f_n[non_harm], freq_res)
    close_f_groups = [non_harm[group] for group in close_f_groups]  # convert to the right indices
    f_sets = [g[np.arange(p1, p2 + 1)]
              for g in close_f_groups
              for p1 in range(len(g) - 1)
              for p2 in range(p1 + 1, len(g))]

    # make an array of sets of frequencies (now with harmonics) to be investigated for replacement
    close_f_groups = frs.chains_within_rayleigh(f_n, freq_res)
    f_sets_h = [g[np.arange(p1, p2 + 1)]
                for g in close_f_groups
                for p1 in range(len(g) - 1)
                for p2 in range(p1 + 1, len(g))
                if np.any(np.array([g_f in harmonics for g_f in g[np.arange(p1, p2 + 1)]]))]

    # join the two lists, and remember which is which
    harm_sets = np.arange(len(f_sets), len(f_sets) + len(f_sets_h))
    f_sets.extend(f_sets_h)
    remove_sets = np.zeros(0, dtype=np.int_)  # sets of frequencies to replace (by 1 freq)
    used_sets = np.zeros(0, dtype=np.int_)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))

    # determine initial bic
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)
    best_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = best_resid - mdl.linear_curve(time, const, slope, i_chunks)
    n_param = ut.n_parameters(n_chunks, n_sin, n_harm)
    bic_prev = gof.calc_bic(resid, n_param)
    bic_init = bic_prev
    n_prev = -1

    # while frequencies are added to the remove list, continue loop
    while len(remove_sets) > n_prev:
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i in used_sets:
                continue

            # make a model of the removed and new sinusoids and subtract/add it from/to the full sinusoid model
            model_sinusoid_r = mdl.sum_sines(time, f_n[set_i], a_n[set_i], ph_n[set_i])
            resid = best_resid + model_sinusoid_r
            const, slope = mdl.linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= mdl.linear_curve(time, const, slope, i_chunks)

            # extract a single freq to try replacing the set
            if i in harm_sets:
                harm_i = np.array([h for h in set_i if h in harmonics])
                f_i = f_n[harm_i]  # fixed f
                a_i, ph_i = pdg.scargle_ampl_phase(time, resid, f_n[harm_i])
            else:
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                out = extract_local(time, resid, f0=edges[0], fn=edges[1])
                f_i, a_i, ph_i = np.array([out[0]]), np.array([out[1]]), np.array([out[2]])

            # make a model including the new freq
            model_sinusoid_n = mdl.sum_sines(time, f_i, a_i, ph_i)
            resid -= model_sinusoid_n
            const, slope = mdl.linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= mdl.linear_curve(time, const, slope, i_chunks)

            # number of parameters and bic
            n_sin_i = n_sin - sum([len(f_sets[j]) for j in remove_sets]) - len(set_i) + len(f_new) + len(f_i) - n_harm
            n_param = ut.n_parameters(n_chunks, n_sin_i, n_harm)
            bic = gof.calc_bic(resid, n_param)
            if np.round(bic_prev - bic, 2) > 0:
                # do not look at sets with the same freqs as the just removed set anymore
                overlap = [j for j, subset in enumerate(f_sets) if np.any(np.array([k in set_i for k in subset]))]
                used_sets = np.unique(np.append(used_sets, overlap))

                # add to list of removed sets
                remove_sets = np.append(remove_sets, i)

                # remember the new frequency (or the current one if it is a harmonic)
                f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
                best_resid += model_sinusoid_r - model_sinusoid_n
                bic_prev = bic

    # lastly re-determine slope and const
    const, slope = mdl.linear_pars(time, best_resid, i_chunks)

    # finally, remove all the designated sinusoids from the lists and add the new ones
    i_to_remove = [k for i in remove_sets for k in f_sets[i]]
    f_n = np.append(np.delete(f_n, i_to_remove), f_new)
    a_n = np.append(np.delete(a_n, i_to_remove), a_new)
    ph_n = np.append(np.delete(ph_n, i_to_remove), ph_new)

    if logger is not None:
        logger.extra(f"Frequency sets replaced: {len(remove_sets)} ({len(i_to_remove)} frequencies). "
                     f"N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f})")

    return const, slope, f_n, a_n, ph_n


def reduce_sinusoids(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt to reduce the number of frequencies taking into account any harmonics if present.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves

    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency. Special attention
    is given to frequencies that are within the Rayleigh criterion of each other.
    It is attempted to replace these by a single frequency.
    """
    # first check if any frequency can be left out (after the fit, this may be possible)
    out_a = remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=logger)
    const, slope, f_n, a_n, ph_n = out_a

    # Now go on to trying to replace sets of frequencies that are close together
    out_b = replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=logger)
    const, slope, f_n, a_n, ph_n = out_b

    return const, slope, f_n, a_n, ph_n


def select_sinusoids(time, flux, flux_err, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Selects the credible frequencies from the given set

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
        If the sinusoids exclude the eclipse model,
        this should be the residuals of the eclipse model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days.
        May be zero.
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        passed_sigma: numpy.ndarray[bool]
            Non-harmonic frequencies that passed the sigma check
        passed_snr: numpy.ndarray[bool]
            Non-harmonic frequencies that passed the signal-to-noise check
        passed_both: numpy.ndarray[bool]
            Non-harmonic frequencies that passed both checks
        passed_harmonic: numpy.ndarray[bool]
            Harmonic frequencies that passed

    Notes
    -----
    Harmonic frequencies that are said to be passing the criteria
    are in fact passing the criteria for individual frequencies,
    not those for a set of harmonics (which would be a looser constraint).
    """
    t_tot = np.ptp(time)
    freq_res = 1.5 / t_tot  # Rayleigh criterion

    # obtain the errors on the sine waves (depends on residual and thus model)
    model_lin = mdl.linear_curve(time, const, slope, i_chunks)
    model_sin = mdl.sum_sines(time, f_n, a_n, ph_n)
    residuals = flux - (model_lin + model_sin)
    errors = ut.formal_uncertainties(time, residuals, flux_err, a_n, i_chunks)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = errors

    # find the insignificant frequencies
    remove_sigma = frs.remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3, sigma_f=3)

    # apply the signal-to-noise threshold
    noise_at_f = pdg.scargle_noise_at_freq(f_n, time, residuals, window_width=1.0)
    remove_snr = frs.remove_insignificant_snr(time, a_n, noise_at_f)

    # frequencies that pass sigma criteria
    passed_sigma = np.ones(len(f_n), dtype=bool)
    passed_sigma[remove_sigma] = False

    # frequencies that pass S/N criteria
    passed_snr = np.ones(len(f_n), dtype=bool)
    passed_snr[remove_snr] = False

    # passing both
    passed_both = (passed_sigma & passed_snr)

    # candidate harmonic frequencies
    passed_harmonic = np.zeros(len(f_n), dtype=bool)
    if p_orb != 0:
        harmonics, harmonic_n = frs.select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol=freq_res / 2, sigma_f=3)
        passed_harmonic[harmonics] = True
    else:
        harmonics = np.array([], dtype=int)
    if logger is not None:
        logger.extra(f"Number of frequencies passed criteria: {np.sum(passed_both)} of {len(f_n)}. "
                     f"Candidate harmonics: {np.sum(passed_harmonic)}, "
                     f"of which {np.sum(passed_both[harmonics])} passed.")

    return passed_sigma, passed_snr, passed_both, passed_harmonic


def refine_orbital_period(p_orb, time, f_n):
    """Find the most likely eclipse period from a sinusoid model

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves

    Returns
    -------
    float
        Orbital period of the eclipsing binary in days

    Notes
    -----
    Uses the sum of distances between the harmonics and their
    theoretical positions to refine the orbital period.

    Period precision is 0.00001, but accuracy is a bit lower.

    Same refine algorithm as used in find_orbital_period
    """
    freq_res = 1.5 / np.ptp(time)  # Rayleigh criterion
    f_nyquist = 1 / (2 * np.min(np.diff(time)))  # nyquist frequency

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    p_orb = 1 / f_refine[mask_peak][i_min_dist]

    return p_orb


def find_orbital_period(time, flux, f_n):
    """Find the most likely eclipse period from a sinusoid model

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves

    Returns
    -------
    tuple
        A tuple containing the following elements:
        p_orb: float
            Orbital period of the eclipsing binary in days
        multiple: float
            Multiple of the initial period that was chosen

    Notes
    -----
    Uses a combination of phase dispersion minimisation and
    Lomb-Scargle periodogram (see Saha & Vivas 2017), and some
    refining steps to get the best period.

    Also tests various multiples of the period.
    Precision is 0.00001 (one part in one-hundred-thousand).
    (accuracy might be slightly lower)
    """
    freq_res = 1.5 / np.ptp(time)  # Rayleigh criterion
    f_nyquist = 1 / (2 * np.min(np.diff(time)))  # nyquist frequency

    # first to get a global minimum do combined PDM and LS, at select frequencies
    periods, phase_disp = pdg.phase_dispersion_minimisation(time, flux, f_n, local=False)
    ampls, _ = pdg.scargle_ampl_phase(time, flux, 1 / periods)
    psi_measure = ampls / phase_disp

    # also check the number of harmonics at each period and include into best f
    n_harm, completeness, distance = frs.harmonic_series_length(1 / periods, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness

    # select the best period, refine it and check double P
    p_orb = periods[np.argmax(psi_h_measure)]

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    p_orb = 1 / f_refine[mask_peak][i_min_dist]

    # reduce the search space by taking limits in the distance metric
    f_left = f_refine[mask_peak][:i_min_dist]
    f_right = f_refine[mask_peak][i_min_dist:]
    d_left = distance_r[mask_peak][:i_min_dist]
    d_right = distance_r[mask_peak][i_min_dist:]
    d_max = np.max(distance_r)
    if np.any(d_left > d_max / 2):
        f_l_bound = f_left[d_left > d_max / 2][-1]
    else:
        f_l_bound = f_refine[mask_peak][0]
    if np.any(d_right > d_max / 2):
        f_r_bound = f_right[d_right > d_max / 2][0]
    else:
        f_r_bound = f_refine[mask_peak][-1]
    bound_interval = f_r_bound - f_l_bound

    # decide on the multiple of the period
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    completeness_p = (len(harmonics) / (f_nyquist // (1 / p_orb)))
    completeness_p_l = (len(harmonics[harmonic_n <= 15]) / (f_nyquist // (1 / p_orb)))

    # check these (commonly missed) multiples
    n_multiply = np.array([1/2, 2, 3, 4, 5])
    p_multiples = p_orb * n_multiply
    n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(1 / p_multiples, f_n, freq_res, f_nyquist)
    h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # if there are very high numbers, add double that fraction for testing
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    if np.any(test_frac[2:] > 3):
        n_multiply = np.append(n_multiply, [2 * n_multiply[2:][test_frac[2:] > 3]])
        p_multiples = p_orb * n_multiply
        n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(1 / p_multiples, f_n, freq_res,
                                                                                f_nyquist)
        h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # compute diagnostic fractions that need to meet some threshold
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    compl_frac = completeness_r_m / completeness_p

    # doubling the period may be done if the harmonic filling factor below f_16 is very high
    f_cut = np.max(f_n[harmonics][harmonic_n <= 15])
    f_n_c = f_n[f_n <= f_cut]
    n_harm_r_2, completeness_r_2, distance_r_2 = frs.harmonic_series_length(1 / p_multiples, f_n_c, freq_res, f_nyquist)
    compl_frac_2 = completeness_r_2[1] / completeness_p_l

    # empirically determined thresholds for the various measures
    minimal_frac = 1.1
    minimal_compl_frac = 0.85
    minimal_frac_low = 0.95
    minimal_compl_frac_low = 0.95

    # test conditions
    test_condition = (test_frac > minimal_frac)
    compl_condition = (compl_frac > minimal_compl_frac)
    test_condition_2 = (test_frac[1] > minimal_frac_low)
    compl_condition_2 = (compl_frac_2 > minimal_compl_frac_low)
    if np.any(test_condition & compl_condition) | (test_condition_2 & compl_condition_2):
        if np.any(test_condition & compl_condition):
            i_best = np.argmax(test_frac[compl_condition])
            p_orb = p_multiples[compl_condition][i_best]
        else:
            p_orb = 2 * p_orb

        # make new bounds for refining
        f_left_b = 1 / p_orb - (bound_interval / 2)
        f_right_b = 1 / p_orb + (bound_interval / 2)

        # refine by using a dense sampling and the harmonic distances
        f_refine_2 = np.arange(f_left_b, f_right_b, 0.00001 / p_orb)
        n_harm_r2, completeness_r2, distance_r2 = frs.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
        h_measure_2 = n_harm_r2 * completeness_r2  # compute h_measure for constraining a domain
        mask_peak = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
        i_min_dist = np.argmin(distance_r2[mask_peak])
        p_orb = 1 / f_refine_2[mask_peak][i_min_dist]

    return p_orb
