"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the data class for handling the user defined data to analyse.

Code written by: Luc IJspeert
"""
import os
import datetime
import h5py
import numpy as np

from star_shine.core import utility as ut
from star_shine.core import visualisation as vis
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class Data:
    """A class to handle light curve data.

    Attributes
    ----------
    file_list: list[str]
        List of ascii light curve files or (TESS) data product '.fits' files.
    data_dir: str
        Root directory where the data files are stored.
    target_id: str
        User defined identification integer for the target under investigation.
    data_id: str
        User defined identification for the dataset used.
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values.
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    flux_counts_medians: numpy.ndarray[Any, dtype[float]]
        Median flux counts per chunk.
    t_tot: float
        Total time base of observations.
    t_mean: float
        Time reference (zero) point of the full light curve.
    t_mean_chunk: numpy.ndarray[Any, dtype[float]]
        Time reference (zero) point per chunk.
    t_int: float
        Integration time of observations (taken to be the median time step by default, may be changed).
    p_orb: float
        The orbital period. Set to 0 to search for the best period.
        If the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    f_min: float
        Minimum frequency for extraction and periodograms
    f_max: float
        Maximum frequency for extraction and periodograms
    """

    def __init__(self, target_id='', data_id=''):
        """Initialises the Data object.

        The data is loaded from the given file(s) and some basic processing is done.
        Either a file name, or target id plus file list must be given.

        Parameters
        ----------
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        """
        self.file_list = []
        self.data_dir = ''
        self.target_id = target_id
        self.data_id = data_id

        # initialise attributes before they are assigned values
        self.time = np.zeros((0,), dtype=np.float_)
        self.flux = np.zeros((0,), dtype=np.float_)
        self.flux_err = np.zeros((0,), dtype=np.float_)
        self.i_chunks = np.zeros((0, 2), dtype=np.int_)
        self.flux_counts_medians = np.zeros((0,), dtype=np.float_)
        self.t_tot = 0.
        self.t_mean = 0.
        self.t_mean_chunk = np.zeros((0,), dtype=np.float_)
        self.t_int = 0.

        self.p_orb = 0.
        self.f_min = 0.
        self.f_max = 0.

        return

    def _check_file_existence(self):
        """Checks whether the given file(s) exist.

        Removes missing files from the file list

        Returns
        -------
        None
        """
        # check for missing files in the list
        missing = []
        for i, file in enumerate(self.file_list):
            if not os.path.exists(os.path.join(self.data_dir, file)):
                missing.append(i)

        # log a message if files are missing
        if len(missing) > 0:
            missing_files = [self.file_list[i] for i in missing]

            # add directory to message
            dir_text = ""
            if self.data_dir is not None:
                dir_text = f" in directory {self.data_dir}"
            message = f"Missing files {missing_files}{dir_text}, removing from list."

            if config.verbose:
                print(message)

            # remove the files
            for i in missing:
                del self.file_list[i]

        return None

    def setter(self, **kwargs):
        """Fill in the attributes with data.

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly

        Returns
        -------
        None
        """
        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    @classmethod
    def load_data(cls, file_list, data_dir='', target_id='', data_id=''):
        """Load light curve data from the file list.

        Parameters
        ----------
        data_dir: str, optional
            Root directory where the data files are stored. Added to the file name. If empty, it is loaded from config.
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        file_list: list[str]
            List of ascii light curve files or (TESS) data product '.fits' files. Exclude the path given to 'data_dir'.
            If only one file is given, its file name is used for target_id (if 'none').

        Returns
        -------
        Data
            Instance of the Data class with the loaded data.
        """
        instance = cls()

        # set the file list and data directory
        if data_dir == '':
            data_dir = config.data_dir
        instance.setter(file_list=file_list, data_dir=data_dir)

        # guard against empty list
        if len(file_list) == 0:
            if config.verbose:
                print("Empty file list provided.")
            return

        # Check if the file(s) exist(s)
        instance._check_file_existence()
        if len(instance.file_list) == 0:
            if config.verbose:
                print("No existing files in file list")
            return

        # set IDs
        if target_id == '':
            target_id = os.path.splitext(os.path.basename(file_list[0]))[0]  # file name is used as identifier
        instance.setter(target_id=target_id, data_id=data_id)

        # add data_dir for loading files, if not None
        if instance.data_dir is None:
            file_list_dir = instance.file_list
        else:
            file_list_dir = [os.path.join(instance.data_dir, file) for file in instance.file_list]

        # load the data from the list of files
        lc_data = ut.load_light_curve(file_list_dir, apply_flags=config.apply_q_flags)
        instance.setter(time=lc_data[0], flux=lc_data[1], flux_err=lc_data[2], i_chunks=lc_data[3], medians=lc_data[4])

        # check for overlapping time stamps
        if np.any(np.diff(instance.time) <= 0):
            if config.verbose:
                print("The time array chunks include overlap.")

        # set derived attributes
        instance.t_tot = np.ptp(instance.time)
        instance.t_mean = np.mean(instance.time)
        instance.t_mean_chunk = np.array([np.mean(instance.time[ch[0]:ch[1]]) for ch in instance.i_chunks])
        instance.t_int = np.median(np.diff(instance.time))  # integration time, taken to be the median time step

        instance.f_min = 0.01 / instance.t_tot
        instance.f_max = ut.frequency_upper_threshold(instance.time, func='min')

        return instance

    @classmethod
    def load(cls, file_name, h5py_file_kwargs):
        """Load a data file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the data from
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.

        Returns
        -------
        Data
            Instance of the Data class with the loaded data.
        """
        # to avoid dict in function defaults
        if h5py_file_kwargs is None:
            h5py_file_kwargs = {}

        # add everything to a dict
        data_dict = {}

        # load the results from the file
        with h5py.File(file_name, 'r', **h5py_file_kwargs) as file:
            # file description
            data_dict['target_id'] = file.attrs['target_id']
            data_dict['data_id'] = file.attrs['data_id']
            data_dict['description'] = file.attrs['description']
            data_dict['date_time'] = file.attrs['date_time']

            # original list of files
            data_dict['data_dir'] = file.attrs['data_dir']
            data_dict['file_list'] = np.copy(file['file_list'])

            # summary statistics
            data_dict['t_tot'] = file.attrs['t_tot']
            data_dict['t_mean'] = file.attrs['t_mean']
            data_dict['t_int'] = file.attrs['t_int']
            data_dict['p_orb'] = file.attrs['p_orb']

            # the time series data
            data_dict['time'] = np.copy(file['time'])
            data_dict['flux'] = np.copy(file['flux'])
            data_dict['flux_err'] = np.copy(file['flux_err'])

            # additional information
            data_dict['i_chunks'] = np.copy(file['i_chunks'])
            data_dict['flux_counts_medians'] = np.copy(file['flux_counts_medians'])
            data_dict['t_mean_chunk'] = np.copy(file['t_mean_chunk'])

        # initiate the Results instance and fill in the results
        instance = cls()
        instance.setter(**data_dict)

        if config.verbose:
            print(f"Loaded data file with target identifier: {data_dict['target_id']}, "
                  f"created on {data_dict['date_time']}. \n"
                  f"Data identifier: {data_dict['data_id']}. \n")

        return instance

    def save(self, file_name):
        """Save the data to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the data to

        Returns
        -------
        None
        """
        # file name must have hdf5 extension
        ext = os.path.splitext(os.path.basename(file_name))[1]
        if ext != '.hdf5':
            file_name = file_name.replace(ext, '.hdf5')

        # save to hdf5
        with h5py.File(file_name, 'w') as file:
            file.attrs['target_id'] = self.target_id
            file.attrs['data_id'] = self.data_id
            file.attrs['description'] = 'Star Shine data file'
            file.attrs['date_time'] = str(datetime.datetime.now())

            # original list of files
            file.attrs['data_dir'] = self.data_dir  # original data directory
            file.create_dataset('file_list', data=self.file_list)
            file['file_list'].attrs['description'] = 'original list of files for the creation of this data file'

            # summary statistics
            file.attrs['t_tot'] = self.t_tot  # Total time base of observations
            file.attrs['t_mean'] = self.t_mean  # Time reference (zero) point of the full light curve
            file.attrs['t_int'] = self.t_int  # Integration time of observations (median time step by default)
            file.attrs['p_orb'] = self.p_orb  # orbital period, if applicable

            # the time series data
            file.create_dataset('time', data=self.time)
            file['time'].attrs['unit'] = 'time unit of the data (often days)'
            file['time'].attrs['description'] = 'timestamps of the observations'
            file.create_dataset('flux', data=self.flux)
            file['flux'].attrs['unit'] = 'median normalised flux'
            file['flux'].attrs['description'] = 'normalised flux measurements of the observations'
            file.create_dataset('flux_err', data=self.flux_err)
            file['flux_err'].attrs['unit'] = 'median normalised flux'
            file['flux_err'].attrs['description'] = 'normalised error measurements in the flux'

            # additional information
            file.create_dataset('i_chunks', data=self.i_chunks)
            file['i_chunks'].attrs['description'] = 'pairs of indices indicating time chunks of the data'
            file.create_dataset('flux_counts_medians', data=self.flux_counts_medians)
            file['flux_counts_medians'].attrs['unit'] = 'raw flux counts'
            file['flux_counts_medians'].attrs['description'] = 'median flux level per time chunk'
            file.create_dataset('t_mean_chunk', data=self.t_mean_chunk)
            file['t_mean_chunk'].attrs['unit'] = 'time unit of the data (often days)'
            file['t_mean_chunk'].attrs['description'] = 'time reference (zero) point of the each time chunk'

        return None

    def plot_light_curve(self, file_name=None, show=True):
        """Plot the light curve data.

        Parameters
        ----------
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot

        Returns
        -------
        None
        """
        vis.plot_lc(self.time, self.flux, self.flux_err, self.i_chunks, file_name=file_name, show=show)
        return None

    def plot_periodogram(self, plot_per_chunk=False, file_name=None, show=True):
        """Plot the light curve data.

        Parameters
        ----------
        plot_per_chunk: bool
            If True, plot the periodogram of all time chunks in one plot.
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot

        Returns
        -------
        None
        """
        vis.plot_pd(self.time, self.flux, self.i_chunks, plot_per_chunk=plot_per_chunk, file_name=file_name, show=show)
        return None
