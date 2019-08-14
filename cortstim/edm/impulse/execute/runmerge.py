import os

import numpy as np
from natsort import natsorted

from cortstim.edm.fragility.execute.basepipe import BasePipe
from cortstim.edp.utils.utils import walk_up_folder


class RunMergeModel(BasePipe):
    """
    A Pipeline class for merging data that is computed on separate (possibly overlapping) windows of data.
    It is a post-merging step after parallelized processing of data.

    It merges together either:
        1. ltvn model
        2. min-2norm perturbation model

    Attributes
    ----------
    tempdir : os.PathLike
        Window size of the data that is passed in.
    numwins : int
        Step size of the data that will be generated

    Notes
    -----

    Depends on the file name pattern that is used in saving algorithm output.

    Examples
    --------
    >>> import os
    >>> from cortstim.edm.fragility.execute.runmerge import RunMergeModel
    >>> model_params = {
    ...     'tempdir': os.path.join(),
    ...     'numwins': 2459,
    ...     }
    >>> modelmerger = RunMergeModel(**model_params)
    >>> modelmerger,mergefragilitydata(outputfilename="")
    """

    def __init__(self, tempdir, numwins):

        self.tempdir = tempdir
        self.numwins = numwins

        # get the list of all the files in natural order
        alltempfiles = [f for f in os.listdir(
            self.tempdir) if f.endswith('.npz') if not f.startswith('.')]
        self.alltempfiles = natsorted(alltempfiles)

    @property
    def numcompletedfiles(self):
        return len(self.alltempfiles)

    def loadmetafile(self, metafilename):
        self.metadata = super(RunMergeModel, self)._loadjsonfile(metafilename)

    def loadmetadata(self, metadata, kwargs=dict()):
        self.metadata = metadata

        for arg in kwargs.keys():
            self.metadata[arg] = kwargs[arg]

    def writemetadata_tofile(self, outputmetafilename):
        self._writejsonfile(self.metadata, outputmetafilename)

    def mergefragilitydata(self, outputfilename):
        """
        Function to merge fragility computed data (i.e. LTVN and Perturbation model)
        Performs a loop over files in the self.tempdir

        :param outputfilename: (os.PathLike) output filepath to save the resulting data
        as .npz file type.
        :return:
        """
        saveflag = True

        # self.logger.info("Merging windowed analysis compute together of "
        #                  "{} windows!".format(len(self.alltempfiles)))

        # save adjacency matrix
        for idx, tempfile in enumerate(self.alltempfiles):
            # get the window numbr of this file just to check
            buff = tempfile.split('_')
            winnum = int(buff[-1].split('.')[0])
            if winnum != idx:
                raise ValueError(
                    "Win num {} should match idx {}".format(winnum, idx))

            tempfilename = os.path.join(self.tempdir, tempfile)
            # load result data
            try:
                data = super(RunMergeModel, self)._loadnpzfile(tempfilename)
            except:
                saveflag = False

                # remove that tempfilename
                os.remove(tempfilename)
                print("File Removed {}!".format(tempfilename))

                # remove success file
                outfile = os.path.basename(outputfilename)
                success_flag_name = os.path.join(walk_up_folder(self.tempdir, 2),
                                                 outfile.replace('fragmodel.npz', 'frag_success.txt'))

                try:
                    os.remove(success_flag_name)
                    print("File Removed {}!".format(success_flag_name))
                except Exception as e:
                    print(e)
                    print("Success flag file already removed!")
                    # self.logger.error("Success flag file already removed at "
                    #                   "{} window!".format(idx))


                continue

            try:
                pertmat = data['pertmat']
                # delfreqs = pertmat_data['delfreqs']
                delvecs = data['delvecs']
                adjmat = data['adjmat']
            except:
                saveflag = False

                # remove that tempfilename
                os.remove(tempfilename)
                print("File Removed {}!".format(tempfilename))

                # remove success file
                outfile = os.path.basename(outputfilename)
                success_flag_name = os.path.join(walk_up_folder(self.tempdir, 2),
                                                 outfile.replace('fragmodel.npz', 'frag_success.txt'))

                try:
                    os.remove(success_flag_name)
                    print("File Removed {}!".format(success_flag_name))
                except Exception as e:
                    print(e)
                    print("Success flag file already removed!")
                    # self.logger.error("Success flag file already removed at "
                    #                   "{} window!".format(idx))

                continue

            if idx == 0:
                numchans, _ = pertmat.shape
                # initializ adjacency matrices over time
                pertmats = np.zeros((numchans, len(self.alltempfiles)))
                delvecs_array = np.zeros(
                    (numchans, numchans, len(self.alltempfiles)), dtype='complex')
                adjmats = np.zeros(
                    (len(self.alltempfiles), numchans, numchans))

                # delfreqs_array = np.zeros((numchans, len(alltempfiles)))
            pertmats[:, idx] = pertmat.ravel()
            delvecs_array[:, :, idx] = delvecs
            adjmats[idx, :, :] = adjmat
            # delfreqs_array[:,idx] = delfreqs

        if saveflag:
            # save adjmats, pertmats and delvecs array along with metadata
            super(RunMergeModel, self)._writenpzfile(outputfilename, adjmats=adjmats,
                                                     pertmats=pertmats,
                                                     delvecs=delvecs_array)
            # self.logger.info("Saved merged analysis together! Dataset includes: "
            #                  "ltvn model: {} "
            #                  "pert model: {} ".format(adjmats.shape, pertmats.shape))

    def mergepertdata(self, outputfilename):
        # save adjacency matrix
        for idx, tempfile in enumerate(self.alltempfiles):
            # get the window numbr of this file just to check
            buff = tempfile.split('_')
            winnum = int(buff[-1].split('.')[0])
            if winnum != idx:
                raise ValueError(
                    "Win num {} should match idx {}".format(winnum, idx))

            tempfilename = os.path.join(self.tempdir, tempfile)
            # load result data
            data = super(RunMergeModel, self)._loadnpzfile(tempfilename)

            pertmat = data['pertmat']
            # delfreqs = pertmat_data['delfreqs']
            delvecs = data['delvecs']

            if idx == 0:
                numchans, _ = pertmat.shape
                # initializ adjacency matrices over time
                pertmats = np.zeros((numchans, len(self.alltempfiles)))
                delvecs_array = np.zeros(
                    (numchans, numchans, len(self.alltempfiles)), dtype='complex')
                # delfreqs_array = np.zeros((numchans, len(alltempfiles)))
            pertmats[:, idx] = pertmat.ravel()
            delvecs_array[:, :, idx] = delvecs
            # delfreqs_array[:,idx] = delfreqs

        # save adjmats along with metadata
        super(RunMergeModel, self)._writenpzfile(outputfilename, pertmats=pertmats,
                                                 delvecs=delvecs_array)

    def mergemvardata(self, outputfilename):
        # save adjacency matrix
        for idx, tempfile in enumerate(self.alltempfiles):
            # get the window numbr of this file just to check
            buff = tempfile.split('_')
            winnum = int(buff[-1].split('.')[0])
            if winnum != idx:
                raise ValueError(
                    "Win num {} should match idx {}".format(winnum, idx))

            tempfilename = os.path.join(self.tempdir, tempfile)
            # load result data
            data = super(RunMergeModel, self)._loadnpzfile(tempfilename)
            adjmat = data['adjmat']
            if idx == 0:
                numchans, _ = adjmat.shape
                # initializ adjacency matrices over time
                adjmats = np.zeros(
                    (len(self.alltempfiles), numchans, numchans))

            adjmats[idx, :, :] = adjmat

        # save adjmats along with metadata
        super(RunMergeModel, self)._writenpzfile(
            outputfilename, adjmats=adjmats)
