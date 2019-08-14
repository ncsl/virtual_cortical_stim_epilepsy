import os

import numpy as np

from cortstim.edp.loaders.base.baseresultloader import BaseResultsLoader
from cortstim.edp.objects.dataset.ltv_object import LtvResult
from cortstim.edp.objects.dataset.perturbationresult_object import ImpulseResult


class ResultLoader(BaseResultsLoader):
    """
    A class for loading in result files from model analysis.
    jsonfilepath is a filepath to some dataset's model computation.

    Attributes
    ----------
    root_dir : os.PathLike
        The root directory that datasets will be located.

    jsonfilepath : os.PathLike
        The filepath to the .json file for the scalp EEG recording

    preload : bool
        Should the class run a loading pipeline for the dataset automatically?

    datatype : str
        The data type of the resulting computation. Supports impulse, freq, impulse, coh

    storagetype : str
        The file storage scheme used for this resulting computation. Supports numpy, mat and hdf.

    Notes
    -----

    In future work, we want to include better referencing schemes.

    Examples
    --------
    >>> from cortstim.edp.loaders.dataset.result.resultloader import ResultLoader
    >>> jsonfilepath = ""
    >>> root_dir = ""
    >>> loader = ResultLoader(jsonfilepath=jsonfilepath,
    ...                root_dir=root_dir, preload=True,
    ...                 datatype='impulse', storagetype='numpy')
    >>> # or
    >>> loader = ResultLoader(root_dir=root_dir)
    >>> jsonfilepaths = loader.jsonfilepaths
    >>> loader.loadpipeline(jsonfilepaths[0])
    """

    def __init__(self, results_dir,
                 datatype='impulse',
                 jsonfilepath=None,
                 preload=False,
                 storagetype='numpy'):
        super(ResultLoader, self).__init__(jsonfilepath=jsonfilepath,
                                           results_dir=results_dir)

        self.datatype = datatype
        self.storagetype = storagetype

        # load in the data
        if preload and self.jsonfilepath is not None:
            self.loadpipeline(self.jsonfilepath)

    def loadpipeline(self, jsonfilepath=None):
        """
        Pipeline loading function that performs the necessary loading functions on a passed in .json filepath.

        :return: model (ModelResult) data object of the model computation.
        """
        if not self.is_loaded:
            if jsonfilepath is None:
                jsonfilepath = self.jsonfilepath
            if self.results_dir not in jsonfilepath:
                jsonfilepath = os.path.join(self.results_dir, jsonfilepath)
            self.jsonfilepath = jsonfilepath

            # load in metadata json object
            self.metadata = self._loadjsonfile(self.jsonfilepath)

            # grab relevant metadata from our metadata
            self.load_metadata(self.metadata)

            # extract the filename for the actual dataset
            resultfilepath = os.path.join(self.results_dir, self.resultfilename)

            print("Loading results data from: ", resultfilepath)
            self.resultstruct = self.load_data(
                resultfilepath, storagetype=self.storagetype)

            # extract data from the model
            model = self.extract_data(datatype=self.datatype)
            self.result = model
            return model
        else:
            raise RuntimeError("Result loader has already been used! Run .reset() to "
                               "reset loader state.")

    def load_data(self, filepath, storagetype='numpy'):
        """
        Generalize loading function that loads in the data if it is .npz, .json, or .hdf.

        :param filepath: (os.PathLike) is the filepath to the data
        :param storagetype: (str) the storage type that is used. Supports numpy, json and hdf.
        :return: resultstruct (ModelResult) the resulting model data structure.
        """
        if storagetype == 'numpy':
            resultstruct = self._loadnpzfile(filepath)
        elif storagetype == 'json':
            resultstruct = self._loadjsonfile(filepath)
        elif storagetype == 'hdf':
            resultstruct = self._loadhd5file(filepath)
        return resultstruct

    def extract_data(self, datatype='impulse'):
        """
        Generalized extraction of data function for the loaded in model dataset.

        :param datatype: (str)
        :return: model (ModelResult) the model that is loaded
        """

        if datatype == 'ltv':
            self.ltvmat = self.resultstruct['adjmats']
            model = LtvResult(adjmats=self.ltvmat, metadata=self.metadata)

        elif datatype == 'impulse':
            self.impulse_l2norms = self.resultstruct['impulse_results']
            model = ImpulseResult(
                impulse_responses=self.impulse_l2norms, metadata=self.metadata)
        else:
            raise RuntimeError("wtf")

        return model

    def split_cez_oez(self, threshold=0.7, offset_sec=10):
        # apply thresholding
        self.result.apply_thresholding_smoothing(threshold=threshold)

        # get channels separated data
        clinonset_map = self.result.get_data()[self.result.cezinds]
        others_map = self.result.get_data()[self.result.oezinds]

        # trim dataset in time
        clinonset_map = self.result.trim_aroundseizure(
            offset_sec=offset_sec, mat=clinonset_map)
        others_map = self.result.trim_aroundseizure(
            offset_sec=offset_sec, mat=others_map)

        clinonset_map = np.mean(clinonset_map, axis=0)
        others_map = np.mean(others_map, axis=0)

        return clinonset_map, others_map
