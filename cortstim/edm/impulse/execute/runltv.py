import contextlib
import multiprocessing as mp
import os
import sys
import traceback
import warnings

import numpy as np

import cortstim.base.config.model_constants as constants
from cortstim.edm.fragility.execute.basepipe import BasePipe
from cortstim.edm.fragility.model.mvarmodel import MvarModel


def get_tempfilename(x): return "temp_{}.npz".format(x)


def mvarjobwin(win):
    # 1: fill matrix of all channels' next EEG data over window
    eegwin = shared_raweeg[:, (shared_samplepoints[win, 0]): (
        shared_samplepoints[win, 1] + 1)]

    # 2: call the function to create the mvar model
    adjmat = shared_mvarmodel.mvaradjacencymatrix(eegwin)

    tempfilename = os.path.join(shared_tempdir, get_tempfilename(win))
    try:
        np.savez_compressed(tempfilename, adjmat=adjmat)
    except BaseException:
        sys.stdout.write(traceback.format_exc())
        return 0
    return 1


def mvarjobnorm(win):
    # 1: fill matrix of all channels' next EEG data over window
    eegwin = shared_raweeg[:, (shared_timepoints[win, 0]): (
        shared_timepoints[win, 1] + 1)]
    # normalize eeg
    eegwin = _normalizets(eegwin)
    eegwin = (eegwin - np.mean(eegwin, axis=1)
              [:, np.newaxis]) / np.std(eegwin, axis=1)[:, np.newaxis]

    # 2: call the function to create the mvar model
    adjmat = shared_mvarmodel.mvaradjacencymatrix(eegwin)

    tempfilename = os.path.join(shared_tempdir, get_tempfilename(win))
    try:
        np.savez_compressed(tempfilename, adjmat=adjmat)
    except BaseException:
        sys.stdout.write(traceback.format_exc())
        return 0
    return 1


class RunLTVModel(BasePipe):
    def __init__(self, winsize=constants.WINSIZE_LTV,
                 stepsize=constants.STEPSIZE_LTV,
                 samplerate=None,
                 **kwargs):
        if samplerate is None:
            raise AttributeError("Please pass in a samplerate \
                from the data")

        self.rawdata = None
        self.tempdir = None

        self.winsize = winsize
        self.stepsize = stepsize
        self.samplerate = samplerate

        self.mvarmodel = MvarModel(
            **kwargs
        )

        # determine if windows have been computed yet
        self.winscomputed = False

    def load_data(self, data):
        self.rawdata = data
        # get number of channels and samples in the raw data
        self.numchans, self.numsignals = self.rawdata.shape
        # compute time and sample windows array
        self._countwindows(self.numsignals)

    def _countwindows(self, numsignals):
        self.compute_samplepoints(numsignals)
        self.compute_timepoints()
        self.timepoints = self.timepoints
        self.samplepoints = self.samplepoints
        self.numwins = self.numwins
        self.winscomputed = True

    def runwindow(self, iwin, normalize=False, save=True):
        assert self.numchans <= self.numsignals
        if save:
            if self.tempdir is None:
                raise AttributeError("You are trying to save resulting computation, \
                    but don't have tempdir set.")

        if self.numchans >= 200:
            warnings.warn(
                "Whoa 200 channels to analyze? Could be too much to compute rn.")

        # compute time and sample windows array
        if self.winscomputed == False:
            self._countwindows(self.numsignals)

        # 1: fill matrix of all channels' next EEG data over window
        win_begin = self.samplepoints[iwin, 0]
        win_end = self.samplepoints[iwin, 1] + 1
        eegwin = self.rawdata[:, win_begin:win_end]

        if normalize:
            sys.stdout.write("NORMALIZING in SINGLMVAR")
            eegwin = (eegwin - np.mean(eegwin, axis=1)
                      [:, np.newaxis]) / np.std(eegwin, axis=1)[:, np.newaxis]

        # 2. Compute the mvar-1 model
        adjmat = self.mvarmodel.mvaradjacencymatrix(eegwin)

        if save:
            tempfilename = os.path.join(self.tempdir, get_tempfilename(iwin))
            np.savez_compressed(tempfilename, adjmat=adjmat)

        return adjmat

    # parallelized functionality
    def init_mp(self):
        global shared_raweeg
        global shared_samplepoints
        global shared_mvarmodel
        global shared_tempdir
        shared_raweeg = self.rawdata
        shared_samplepoints = self.samplepoints
        shared_mvarmodel = self.mvarmodel
        shared_tempdir = self.tempdir

    def runmpall(self, numcores=4, compute_on_missing_wins=False):
        if self.numchans >= 200:
            warnings.warn(
                "Whoa 200 channels to analyze? Could be too much to compute rn.")

        print("about to run pool!")
        # initialize variables needed for multiprocessing run
        self.init_mp()

        # Run multiprocessing pool over indices
        if compute_on_missing_wins:
            missingwins = self.getmissingwins(self.tempdir, self.numwins)
            # Run multiprocessing pool over indices
            with contextlib.closing(mp.Pool(processes=self.numcores)) as p:
                mvarresults = p.map(mvarjobwin, missingwins)
        else:
            # Run multiprocessing pool over indices
            with contextlib.closing(mp.Pool(processes=self.numcores)) as p:
                mvarresults = p.map(mvarjobwin, range(self.numwins))
        p.join()
        sys.stdout.write(
            'Finished mvar model computation for {} windows.'.format(self.numwins))
        return 1
