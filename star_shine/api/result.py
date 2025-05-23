"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the result class for handling the analysis results.

Code written by: Luc IJspeert
"""
import os
import numpy as np

from star_shine.core import model as mdl, frequency_sets as frs, utility as ut
from star_shine.core import io
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class Result:
    """A class to handle analysis results.

    Attributes
    ----------
    target_id: str
        User defined identification number or name for the target under investigation.
    data_id: str
        User defined identification name for the dataset used.
    description: str
        User defined description of the result in question.
    n_param: int
        Number of free parameters in the model.
    bic: float
        Bayesian Information Criterion of the residuals.
    noise_level: float
        The noise level (standard deviation of the residuals).
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves.
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves.
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves.
    c_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the constant for each sector.
    sl_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slope for each sector.
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sine wave.
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitude for each sine wave (these are identical).
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phase for each sine wave.
    c_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the constant for each sector.
    sl_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the slope for each sector.
    f_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the frequency for each sine wave.
    a_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the amplitude for each sine wave (these are identical).
    ph_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the phase for each sine wave.
    passed_sigma: numpy.ndarray[bool]
        Sinusoids that passed the sigma check.
    passed_snr: numpy.ndarray[bool]
        Sinusoids that passed the signal-to-noise check.
    passed_both: numpy.ndarray[bool]
        Sinusoids that passed both checks.
    p_orb: float
        Orbital period.
    p_err: float
        Error in the orbital period.
    p_hdi: numpy.ndarray[2, dtype[float]]
        HDI for the period.
    passed_harmonic: numpy.ndarray[bool]
        Harmonic sinusoids that passed.
    """

    def __init__(self):
        """Initialises the Result object."""
        # descriptive
        self.target_id = ''
        self.data_id = ''
        self.description = ''

        # summary statistics
        self.n_param = -1
        self.bic = -1.
        self.noise_level = -1.

        # linear model parameters
        # y-intercepts
        self.const = np.zeros((0,))
        self.c_err = np.zeros((0,))
        self.c_hdi = np.zeros((0, 2))
        # slopes
        self.slope = np.zeros((0,))
        self.sl_err = np.zeros((0,))
        self.sl_hdi = np.zeros((0, 2))

        # sinusoid model parameters
        self.sinusoid_property_list = ['f_n', 'a_n', 'ph_n']
        self.property_type_list = ['', '_err', '_hdi']
        self.sinusoid_passed_list = ['passed_sigma', 'passed_snr', 'passed_both']
        # note: I could make the below attrs with a nice loop but then my IDE complains about undefined references

        # frequencies
        self.f_n = np.zeros((0,))
        self.f_n_err = np.zeros((0,))
        self.f_n_hdi = np.zeros((0, 2))
        # amplitudes
        self.a_n = np.zeros((0,))
        self.a_n_err = np.zeros((0,))
        self.a_n_hdi = np.zeros((0, 2))
        # phases
        self.ph_n = np.zeros((0,))
        self.ph_n_err = np.zeros((0,))
        self.ph_n_hdi = np.zeros((0, 2))
        # passing criteria
        self.passed_sigma = np.zeros((0,), dtype=bool)
        self.passed_snr = np.zeros((0,), dtype=bool)
        self.passed_both = np.zeros((0,), dtype=bool)

        # harmonic model
        self.p_orb = 0.
        self.p_err = 0.
        self.p_hdi = np.zeros((2,))
        self.passed_harmonic = np.zeros((0,), dtype=bool)

        return

    def setter(self, **kwargs):
        """Fill in the attributes with results.

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly
        """
        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    def get_dict(self):
        """Make a dictionary of the attributes.

        Primarily for saving to file.

        Returns
        -------
        result_dict: dict
            Dictionary of the result attributes and fields
        """
        # make a dictionary of the fields to be saved
        result_dict = dict()
        result_dict['target_id'] = self.target_id
        result_dict['data_id'] = self.data_id
        result_dict['description'] = self.description
        result_dict['date_time'] = ut.datetime_formatted()

        result_dict['n_param'] = self.n_param  # number of free parameters
        result_dict['bic'] = self.bic  # Bayesian Information Criterion of the residuals
        result_dict['noise_level'] = self.noise_level  # standard deviation of the residuals

        # orbital period
        result_dict['p_orb'] = self.p_orb

        # the linear model
        # y-intercepts
        result_dict['const'] = self.const
        result_dict['c_err'] = self.c_err
        result_dict['c_hdi'] = self.c_hdi

        # slopes
        result_dict['slope'] = self.slope
        result_dict['sl_err'] = self.sl_err
        result_dict['sl_hdi'] = self.sl_hdi

        # the sinusoid model
        # frequencies
        result_dict['f_n'] = self.f_n
        result_dict['f_n_err'] = self.f_n_err
        result_dict['f_n_hdi'] = self.f_n_hdi

        # amplitudes
        result_dict['a_n'] = self.a_n
        result_dict['a_n_err'] = self.a_n_err
        result_dict['a_n_hdi'] = self.a_n_hdi

        # phases
        result_dict['ph_n'] = self.ph_n
        result_dict['ph_n_err'] = self.ph_n_err
        result_dict['ph_n_hdi'] = self.ph_n_hdi

        # selection criteria
        result_dict['passed_sigma'] = self.passed_sigma
        result_dict['passed_snr'] = self.passed_snr
        result_dict['passed_both'] = self.passed_both
        result_dict['passed_harmonic'] = self.passed_harmonic

        return result_dict

    @classmethod
    def load(cls, file_name, h5py_file_kwargs=None, logger=None):
        """Load a result file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.
        logger: logging.Logger, optional
            Instance of the logging library.

        Returns
        -------
        Result
            Instance of the Result class with the loaded results.
        """
        # guard for existing file
        if not os.path.isfile(file_name):
            instance = cls()
            return instance

        # add everything to a dict
        result_dict = io.load_result_hdf5(file_name, h5py_file_kwargs=h5py_file_kwargs)

        # initiate the Results instance and fill in the results
        instance = cls()
        instance.setter(**result_dict)

        if logger is not None:
            logger.info(f"Loaded result file with target identifier: {result_dict['target_id']}, "
                        f"created on {result_dict['date_time']}. Data identifier: {result_dict['data_id']}. "
                        f"Description: {result_dict['description']}")

        return instance

    @classmethod
    def load_conditional(cls, file_name, logger=None):
        """Load a result file into a Result instance only if it exists and if no overwriting.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        logger: logging.Logger, optional
            Instance of the logging library.

        Returns
        -------
        Result
            Instance of the Result class with the loaded results.
            Returns empty Result if condition not met.
        """
        # guard for existing file when not overwriting
        if (not os.path.isfile(file_name)) | config.overwrite:
            instance = cls()
            return instance

        # make the Data instance and load the data
        instance = cls.load(file_name, logger=logger)

        return instance

    def save(self, file_name):
        """Save the results to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the results to
        """
        # get a dictionary of the fields to be saved
        result_dict = self.get_dict()

        # io module handles writing to file
        io.save_result_hdf5(file_name, result_dict)

        return None

    def save_as_csv(self, file_name):
        """Write multiple ascii csv files for human readability.

        Parameters
        ----------
        file_name: str
            File name to save the results to
        """
        # get a dictionary of the fields to be saved
        result_dict = self.get_dict()

        # io module handles writing to file
        io.save_result_csv(file_name, result_dict)

        return None

    def save_conditional(self, file_name):
        """Save a result file only if it doesn't exist or if it exists and if no overwriting.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        """
        if (not os.path.isfile(file_name)) | config.overwrite:
            self.save(file_name)

            # save csv files if configured
            if config.save_ascii:
                self.save_as_csv(file_name)

        return None

    def update_n_param(self):
        """Evaluate and set the number of free parameters of the model."""
        # check harmonics
        n_harm = 0
        if self.p_orb > 0:
            harmonics, harmonic_n = frs.find_harmonics_from_pattern(self.f_n, self.p_orb, f_tol=1e-9)
            n_harm = len(harmonics)

        # equation for number of parameters
        n_chunks = len(self.const)
        n_sinusoids = len(self.f_n)
        n_param = ut.n_parameters(n_chunks, n_sinusoids, n_harm)
        self.setter(n_param=n_param)

        return None

    def remove_sinusoids(self, indices):
        """Remove the sinusoids at the provided indices from the list.

        Parameters
        ----------
        indices: numpy.ndarray[Any, dtype[int]]
            Indices of the sinusoids to remove.
        """
        # loop through frequencies, amplitudes, and phases and remove the index values
        for sinusoid_property in self.sinusoid_property_list:
            for property_type in self.property_type_list:
                key = sinusoid_property + property_type
                property_value = getattr(self, key)
                if np.max(indices) < len(property_value):
                    setattr(self, key, np.delete(property_value, indices))

        # passing criteria
        if np.max(indices) < len(self.passed_both):
            self.passed_sigma = np.delete(self.passed_sigma, indices)
            self.passed_snr = np.delete(self.passed_snr, indices)
            self.passed_both = np.delete(self.passed_both, indices)

        return None

    def model_linear(self, time, i_chunks):
        """Returns a piece-wise linear curve for the time series with the current parameters.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The model time series of a (set of) straight line(s)
        """
        curve = mdl.linear_curve(time, self.const, self.slope, i_chunks)

        return curve

    def model_sinusoid(self, time):
        """Returns a sum of sine waves for the time series with the current parameters.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model time series of a sum of sine waves. Varies around 0.
        """
        curve = mdl.sum_sines(time, self.f_n, self.a_n, self.ph_n)

        return curve
