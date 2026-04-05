"""STAR SHADOW
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains functions for the use in and to perform Markov Chain Monte Carlo (MCMC) with PyMC3
"""

import os
import logging
import numpy as np
import scipy as sp
import scipy.stats

try:
    # optional functionality
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    from fastprogress import fastprogress
except (ImportError, AttributeError) as e:
    if e == AttributeError:
        print('PyMC3 functionality unavailable, likely incompatible numpy version')
    pass


LOGICAL_PROC = len(os.sched_getaffinity(0))  # number of available logical processors


def sample_sinusoid_(ts_model, logger=None):
    """NUTS sampling of a linear + sinusoid + eclipse model

    Large phase errors (used as initial sampling parameters) do not play well with the circular distribution
    used in sampling. These are included by limiting their error values. Be mindful of this when interpreting the
    sampling output for the phase errors.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        inf_data: object
            Arviz inference data object
        par_means: list[float]
            Parameter mean values in the following order:
            const, slope, f_n, a_n, ph_n
        par_hdi: list[float]
            Parameter HDI error values, same order as par_means
    """
    # setup
    time_t = ts_model.time.reshape(-1, 1)  # transposed time
    t_mean = pt.as_tensor_variable(ts_model.t_mean)
    t_mean_ch = pt.as_tensor_variable(ts_model.t_mean_chunk)
    lin_shape = (len(ts_model.linear.const),)
    sin_shape = (len(ts_model.sinusoid.f_n),)

    # progress bar
    if logger is not None:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)

    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=ts_model.linear.const, sigma=ts_model.linear.const_err,
                             shape=lin_shape, initval=ts_model.linear.const)
        slope_pm = pm.Normal('slope', mu=ts_model.linear.slope, sigma=ts_model.linear.slope_err,
                             shape=lin_shape, initval=ts_model.linear.slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (ts_model.time[s[0]:s[1]] - t_mean_ch[k])
                         for k, s in enumerate(ts_model.linear.i_chunks)]
        model_linear = pt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=ts_model.sinusoid.f_n, sigma=ts_model.sinusoid.f_n_err, lower=0,
                                    shape=sin_shape, initval=ts_model.sinusoid.f_n)
        a_n_pm = pm.TruncatedNormal('a_n', mu=ts_model.sinusoid.a_n, sigma=ts_model.sinusoid.a_n_err, lower=0,
                                    shape=sin_shape, initval=ts_model.sinusoid.a_n)
        ph_n_pm = pm.VonMises('ph_n', mu=ts_model.sinusoid.ph_n, kappa=1 / ts_model.sinusoid.ph_n_err**2,
                              shape=sin_shape, initval=ts_model.sinusoid.ph_n)
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (time_t - t_mean)) + ph_n_pm), axis=1)
        # full light curve model
        model = model_linear + model_sinusoid
        # observed distribution
        pm.Normal('obs', mu=model, sigma=ts_model.noise_level(), observed=ts_model.flux)

    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=LOGICAL_PROC, progressbar=(logger is not None))
    
    if logger is not None:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])

    # stacked parameter chains
    const_ch = az.extract(inf_data, var_names=["const"]).to_numpy()
    slope_ch = az.extract(inf_data, var_names=["slope"]).to_numpy()
    f_n_ch = az.extract(inf_data, var_names=["f_n"]).to_numpy()
    a_n_ch = az.extract(inf_data, var_names=["a_n"]).to_numpy()
    ph_n_ch = az.extract(inf_data, var_names=["ph_n"]).to_numpy()

    # parameter means
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    par_means = [const_m, slope_m, f_n_m, a_n_m, ph_n_m]

    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)

    # convert interval to error bars
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [const_e, slope_e, f_n_e, a_n_e, ph_n_e]

    return inf_data, par_means, par_hdi


def sample_sinusoid(ts_model, logger=None):
    """NUTS sampling of a linear + sinusoid + harmonic model

    Large phase errors (used as initial sampling parameters) do not play well with the circular distribution
    used in sampling. These are included by limiting their error values. Be mindful of this when interpreting the
    sampling output for the phase errors.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        inf_data: object
            Arviz inference data object
        par_means: list[float]
            Parameter mean values in the following order:
            f_base, const, slope, f_n, a_n, ph_n
        par_hdi: list[float]
            Parameter HDI error values, same order as par_means
    """
    # get the harmonic base frequencies and harmonic parameters
    h_base_unique, h_base_map = ts_model.sinusoid.get_h_base_map()
    harmonics = ts_model.sinusoid.harmonics
    h_mult = ts_model.sinusoid.h_mult[harmonics]  # only include the harmonics

    # indices of harmonic and non-harmonic sinusoids
    i_h = np.arange(len(harmonics))[harmonics]
    i_non_h = np.arange(len(harmonics))[~harmonics]

    # get the sinusoid parameters
    f_n = ts_model.sinusoid.f_n
    a_n = ts_model.sinusoid.a_n
    ph_n = ts_model.sinusoid.ph_n

    # get the sinusoid errors
    f_n_err = ts_model.sinusoid.f_n_err
    a_n_err = ts_model.sinusoid.a_n_err
    ph_n_err = ts_model.sinusoid.ph_n_err

    # get the linear parameters
    const = ts_model.linear.const
    slope = ts_model.linear.slope

    # get the linear errors
    const_err = ts_model.linear.const_err
    slope_err = ts_model.linear.slope_err

    # get the harmonic parameters
    f_base = f_n[h_base_unique]
    f_base_err = f_n_err[h_base_unique]

    # prepare some numbers
    lin_shape = (ts_model.linear.n_chunks,)
    sin_shape = (len(i_non_h),)
    harm_shape = (ts_model.sinusoid.n_harm,)
    n_base_shape = (ts_model.sinusoid.n_base,)

    # setup
    time_t = ts_model.time.reshape(-1, 1)  # transposed time
    t_mean = pt.as_tensor_variable(ts_model.t_mean)
    t_mean_ch = pt.as_tensor_variable(ts_model.t_mean_chunk)

    # progress bar
    if logger is not None:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)

    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=const_err, shape=lin_shape, initval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=slope_err, shape=lin_shape, initval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (ts_model.time[s[0]:s[1]] - t_mean_ch[k])
                         for k, s in enumerate(ts_model.i_chunks)]
        model_linear = pt.concatenate(linear_curves)

        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n[i_non_h], sigma=f_n_err[i_non_h], lower=0, shape=sin_shape,
                                    initval=f_n[i_non_h])
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n[i_non_h], sigma=a_n_err[i_non_h], lower=0, shape=sin_shape,
                                    initval=a_n[i_non_h])
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n[i_non_h], kappa=1 / ph_n_err[i_non_h]**2,
                              shape=sin_shape, initval=ph_n[i_non_h])
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (time_t - t_mean)) + ph_n_pm), axis=1)

        # harmonic parameter models
        f_base_pm = pm.TruncatedNormal('f_base', mu=f_base, sigma=f_base_err, lower=0, shape=n_base_shape,
                                       initval=f_base)
        f_h_pm = pm.Deterministic('f_h', h_mult * f_base_pm[h_base_map])
        a_h_pm = pm.TruncatedNormal('a_h', mu=a_n[i_h], sigma=a_n_err[i_h], lower=0, shape=harm_shape,
                                    initval=a_n[i_h])
        ph_h_pm = pm.VonMises('ph_h', mu=ph_n[i_h], kappa=1 / ph_n_err[i_h]**2, shape=harm_shape,
                              initval=ph_n[i_h])
        # sum of harmonic sinusoids
        model_harmonic = pm.math.sum(a_h_pm * pm.math.sin((2 * np.pi * f_h_pm * (time_t - t_mean)) + ph_h_pm), axis=1)

        # full light curve model
        model = model_linear + model_sinusoid + model_harmonic
        # observed distribution
        pm.Normal('obs', mu=model, sigma=ts_model.noise_level(), observed=ts_model.flux)
    
    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=LOGICAL_PROC, progressbar=(logger is not None))
    
    if logger is not None:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])

    # stacked parameter chains
    const_ch = az.extract(inf_data, var_names=["const"]).to_numpy()
    slope_ch = az.extract(inf_data, var_names=["slope"]).to_numpy()
    f_n_ch = az.extract(inf_data, var_names=["f_n"]).to_numpy()
    a_n_ch = az.extract(inf_data, var_names=["a_n"]).to_numpy()
    ph_n_ch = az.extract(inf_data, var_names=["ph_n"]).to_numpy()
    f_h_ch = az.extract(inf_data, var_names=["f_h"]).to_numpy()
    a_h_ch = az.extract(inf_data, var_names=["a_h"]).to_numpy()
    ph_h_ch = az.extract(inf_data, var_names=["ph_h"]).to_numpy()

    # parameter means
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    f_h_m = h_mult * f_base_pm[h_base_map]  # taking the mean from the chain leads to slightly different results and is wrong
    a_h_m = np.mean(a_h_ch, axis=1).flatten()
    ph_h_m = sp.stats.circmean(ph_h_ch, axis=1).flatten()
    f_n_m = np.append(f_n_m, f_h_m)
    a_n_m = np.append(a_n_m, a_h_m)
    ph_n_m = np.append(ph_n_m, ph_h_m)
    par_means = [const_m, slope_m, f_n_m, a_n_m, ph_n_m]

    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_h_e = az.hdi(np.moveaxis(f_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_h_e = az.hdi(np.moveaxis(a_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_h_e = az.hdi(np.moveaxis(ph_h_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_n_e = np.append(f_n_e, f_h_e, axis=0)
    a_n_e = np.append(a_n_e, a_h_e, axis=0)
    ph_n_e = np.append(ph_n_e, ph_h_e, axis=0)

    # convert interval to error bars
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [const_e, slope_e, f_n_e, a_n_e, ph_n_e]

    return inf_data, par_means, par_hdi
