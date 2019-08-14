import io
import json
import os
import warnings

import numpy as np
from natsort import natsorted

from cortstim.base.config.masterconfig import Config
from cortstim.base.utils.data_structures_utils import NumpyEncoder
from cortstim.base.utils.data_structures_utils import ensure_list
from cortstim.base.utils.log_error import initialize_logger, warning

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class BasePipe(object):
    """
    Base executing class for fragility module.

    Makes sure children classes inherit certain function names.

    Attributes
    ----------
    winsize : int
        Window size of the data that is passed in.
    stepsize : int
        Step size of the data that will be generated
    samplerate : int
        Number of dimensions (this is always 2)

    """

    def __init__(self, winsize, stepsize, samplerate, config=None):
        self.config = config or Config()
        # initializes the logger to output files to FOLDER_LOG
        self.logger = initialize_logger(
            self.__class__.__name__, self.config.out.FOLDER_LOGS)

        if samplerate < 200:
            warning(
                "Sample rate is < 200, which is difficult to say works!", self.logger)

        assert isinstance(winsize, int)
        assert isinstance(stepsize, int)

        # self.model = model
        self.winsize = winsize
        self.stepsize = stepsize
        self.samplerate = samplerate

        # compute the number of samples in window and step
        self._setsampsinwin()
        self._setsampsinstep()

    @property
    def winsize_ms(self):
        return np.divide(self.winsamps, self.samplerate)

    @property
    def stepsize_ms(self):
        return np.divide(self.stepsamps, self.samplerate)

    @property
    def winsize_samples(self):
        return self.winsamps

    @property
    def stepsize_samples(self):
        return self.stepsamps

    @property
    def parameters(self):
        return (self.winsize_samples, self.stepsize_samples, self.samplerate)

    def _setsampsinwin(self):
        self.winsamps = self.winsize
        if self.winsamps % 1 != 0:
            warnings.warn("The number of samples within your window size is not an even integer.\
                          Consider increasing/changing the window size.")

    def _setsampsinstep(self):
        self.stepsamps = self.stepsize
        if self.stepsamps % 1 != 0:
            warnings.warn("The number of samples within your step size is not an even integer.\
                          Consider increasing/changing the step size.")

    def compute_timepoints(self):
        """
        Helper function to compute the corresponding timepoints of each window in terms of
        seconds.

        :return: timepoints (list) a list of the timepoints in seconds of each window begin/end.
        """
        timepoints = self.samplepoints / self.samplerate * 1000
        self.timepoints = timepoints
        return timepoints

    def compute_samplepoints(self, numtimepoints, copy=True):
        """
        Function to compute the index endpoints in terms of signal samples for each sliding
        window in a piped algorithm.

        :param numtimepoints: (int) T in an NxT matrix of raw data.
        :param copy: (bool) should the function return a copy?
        :return: samplepoints (list; optional) list of samplepoint indices that define each window
        of data
        """
        # Creates a [n,2] array that holds the sample range of each window that
        # is used to index the raw data for a sliding window analysis
        samplestarts = np.arange(0,
                                 numtimepoints - self.winsize + 1.,
                                 self.stepsize).astype(int)
        sampleends = np.arange(self.winsize - 1.,
                               numtimepoints,
                               self.stepsize).astype(int)
        samplepoints = np.append(samplestarts[:, np.newaxis],
                                 sampleends[:, np.newaxis], axis=1)
        self.numwins = samplepoints.shape[0]
        if copy:
            self.samplepoints = samplepoints
        else:
            return samplepoints

    def settempdir(self, tempdir):
        """
        Function to set the temporary directory for where to store temporary results of each
        sliding window.

        :param tempdir: (os.PathLike) file path for where to save these temporar results
        :return: None
        """
        self.tempdir = tempdir
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

    def load_data(self, rawdata):
        raise NotImplementedError(
            "Need to implement function for loading data!")

    def runwindow(self):
        raise NotImplementedError("Not implemented error! Need to\
            define a function to run window of data!")

    def initmp(self):
        raise NotImplementedError("Not implemented error! Need to\
            define a function to run window of data!")

    def runmpall(self):
        raise NotImplementedError("Need to implement a function to \
            run for along with multiprocessing!")

    def runarray(self):
        raise NotImplementedError("Need to implement a function to \
            run with array jobs in SLURM.")

    def loadmetadata(self, metadata):
        self.metadata = metadata

    def _loadnpzfile(self, npzfilename):
        if not npzfilename.endswith('.npz'):
            npzfilename += '.npz'

        result = np.load(npzfilename)
        return result

    def _writenpzfile(self, npzfilename, **kwargs):
        if not npzfilename.endswith('.npz'):
            npzfilename += '.npz'

        np.savez_compressed(npzfilename, **kwargs)

    def _writejsonfile(self, metadata, metafilename):
        """
        Helper function to write a dictionary metadata object to a filepath.

        :param metadata: (dict) a dictionary that stores metadata.
        :param metafilename: (os.PathLike) a filepath to where to save the metadata in json format.
        :return: None
        """
        if not metafilename.endswith('.json'):
            metafilename += '.json'
        with io.open(metafilename, mode='w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata,
                              indent=4, sort_keys=True,
                              cls=NumpyEncoder, separators=(',', ': '),
                              ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        """
        Helper function for loading dictionary metadata from a json filepath.
        :param metafilename: (os.PathLike) a filepath to load json data from.
        :return: metadata (dict)
        """
        if not metafilename.endswith('.json'):
            metafilename += '.json'
        try:
            with open(metafilename, mode='r', encoding='utf8') as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except:
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read()
            metadata = json.loads(json_str)
        return metadata

    def getmissingwins(self, tempresultsdir, numwins):
        """
        Function to compute the missing data within the temporary local directory

        :param tempresultsdir: (os.PathLike) where to check for temporary result files.
        :param numwins: (int) the number of windows that is needed to complete the
                            models through the time series data
        :return: winstoanalyze (list) is the list of window indices to analyze
        """
        # get a list of all the sorted files in our temporary directory
        tempfiles = [f for f in os.listdir(
            tempresultsdir) if not f.startswith('.')]
        tempfiles = natsorted(tempfiles)

        if len(tempfiles) == 0:
            return np.arange(0, numwins).astype(int)

        if numwins != len(tempfiles):
            # if numwins does not match, get list of wins not completed
            totalwins = np.arange(0, numwins, dtype='int')
            tempfiles = np.array(tempfiles)[:, np.newaxis]

            # patient+'_'+str(iwin) = the way files are named
            def func(filename):
                return int(
                    filename[0].split('_')[-1].split('.')[0])

            tempwins = np.apply_along_axis(func, 1, tempfiles)
            winstoanalyze = list(set(totalwins) - set(tempwins))
        else:
            winstoanalyze = []
        return ensure_list(winstoanalyze)
