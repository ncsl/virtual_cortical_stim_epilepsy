import os
import sys
from pathlib import Path
import collections

sys.path.append('../../')
sys.path.append('../../../')

from cortstim.edp.loaders.dataset.timeseries.ieegrecording import iEEGRecording
from cortstim.edp.loaders.dataset.timeseries.scalprecording import ScalpRecording
from cortstim.edp.loaders.dataset.result.resultloader import ResultLoader
from cortstim.edp.objects.baseobject import BaseMeta
from cortstim.edp.utils.utils import walk_up_folder

'''
Loading function wrappers for the results data
'''


def load_freq_data(results_dir, outputmetafilepath):
    resultfreq = ResultLoader(results_dir=results_dir,
                              jsonfilepath=outputmetafilepath,
                              datatype='freq',
                              preload=False)
    model = resultfreq.loadpipeline(jsonfilepath=outputmetafilepath)
    return model


def load_corr_data(results_dir, outputmetafilepath):
    result = ResultLoader(results_dir=results_dir,
                          jsonfilepath=outputmetafilepath,
                          datatype='corr',
                          preload=False)
    model = result.loadpipeline(jsonfilepath=outputmetafilepath)
    return model


def load_coh_data(results_dir, outputmetafilepath):
    resultfreq = ResultLoader(results_dir=results_dir,
                              jsonfilepath=outputmetafilepath,
                              datatype='coh',
                              preload=False)
    model = resultfreq.loadpipeline(jsonfilepath=outputmetafilepath)
    return model


def load_results_data(results_dir, outputmetafilepath, datatype='frag'):
    resultfrag = ResultLoader(results_dir=results_dir,
                              jsonfilepath=outputmetafilepath,
                              datatype=datatype,
                              preload=False)
    fragmodel = resultfrag.loadpipeline(jsonfilepath=outputmetafilepath)
    return fragmodel


def load_ltvn_results(results_dir, outputmetafilepath):
    resultfrag = ResultLoader(results_dir=results_dir,
                              jsonfilepath=outputmetafilepath,
                              datatype='ltv',
                              preload=False)
    model = resultfrag.loadpipeline(jsonfilepath=outputmetafilepath)
    return model


'''
Loading function wrappers for raw data
'''


def load_raw_data(root_dir, jsonfilepath, reference='monopolar',
                  remove_wm_contacts=True, apply_mask=True, modality='ieeg'):
    if modality == 'ieeg' or modality == 'seeg':
        recording = iEEGRecording(root_dir=root_dir,
                                  jsonfilepath=jsonfilepath,
                                  apply_mask=apply_mask,
                                  remove_wm_contacts=remove_wm_contacts,
                                  reference=reference)

    elif modality == 'scalp':
        recording = ScalpRecording(root_dir=root_dir,
                                   jsonfilepath=jsonfilepath,
                                   reference=reference)

    eegts = recording.loadpipeline(jsonfilepath)
    print(recording.patient_id, recording.dataset_id)
    # print(recording.onsetsec, recording.offsetsec)
    print("Data shape: ", eegts.shape)
    print("modality: ", modality)
    print(eegts.chanlabels, flush=True)
    return eegts


def pair_name_to_infiles_allcenters(config):
    # get all *.fif files recursively under directory A
    fif_path = Path(config['rawdatadir']).glob("*/*/seeg/fif/*.fif")

    # pair fif name to infile path using a dictionary
    fif_infiles_dict_json = {}

    # go through each raw data structure
    for f in fif_path:
        # get the actual dataset name
        orig_fif_name = f.name
        fif_name = f.name.replace('_raw.fif', '')
        fifdir = os.path.dirname(f)

        # set the dictionary
        fif_infiles_dict_json[fif_name] = str(os.path.join(fifdir,
                                                           orig_fif_name.replace(
                                                               '.fif', '.json').replace(
                                                               '_raw', '')))
    return fif_infiles_dict_json


def pair_name_to_infiles(config, ignore_interictal=False, ignore_ictal=False):
    if ignore_ictal and ignore_interictal:
        raise Exception("Can't ignore both interictal and ictal datasets!")

    # hack inserted to make sure all modalities are for ieeg
    modality = config['modality']
    if modality == 'ieeg' and config['center'] != 'umf':
        modality = 'seeg'

    # get all *.fif files recursively under directory A
    fif_path = Path(os.path.join(config['rawdatadir'],
                                 config['center'])).glob("*/{}/fif/*.fif".format(modality))

    # if not fif_path:
    #     modality = 'ieeg'
    #     fif_path = Path(os.path.join(config['rawdatadir'],
    #                                  config['center'])).glob("*/{}/fif/*.fif".format(modality))

    # pair fif name to infile path using a dictionary
    fif_infiles_dict_json = {}

    # get a list of all the patients
    patient_ids = dict()
    for f in fif_path:
        # get patient ids
        patid = f.name.split("_")[0]
        patient_ids[f.name] = patid
    stored_patients = collections.defaultdict(int)

    # get all *.fif files recursively under directory A
    fif_path = Path(os.path.join(config['rawdatadir'],
                                 config['center'])).glob("*/{}/fif/*.fif".format(modality))

    # go through each raw data structure
    for f in fif_path:
        # get the actual dataset name
        orig_fif_name = f.name
        fif_name = f.name.replace('_raw.fif', '')
        fifdir = os.path.dirname(f)

        if 'jh106' in fif_name:
            continue
        if 'pt17' in fif_name:
            continue
        if 'la01' in fif_name:
            continue

        if ignore_interictal:
            if 'ii' in fif_name:
                continue
        if ignore_ictal:
            if 'sz' in fif_name:
                continue

        # only keep one of each patient id
        # patid = patient_ids.pop(orig_fif_name, None)
        # if stored_patients[patid] >= 2:
        #     continue
        # stored_patients[patid] += 1

        # if not 'tvb' in fif_name:
        #     continue

        # set the dictionary
        fif_infiles_dict_json[fif_name] = str(os.path.join(fifdir,
                                                           orig_fif_name.replace(
                                                               '.fif', '.json').replace(
                                                               '_raw', '')))
        # break
    return fif_infiles_dict_json


def pair_name_to_resultfiles(config, resultsdir, ignore_interictal=True):
    # get all *.npz files recursively under directory A
    file_path = Path(resultsdir).glob("*.npz")

    # pair fif name to infile path using a dictionary
    resultfiles_dict_json = {}

    # go through each raw data structure
    for f in file_path:
        jsonfilename = f.name.replace('fragmodel.npz', 'frag.json')
        if config['modelname'] == 'coherence':
            jsonfilename = f.name.replace("model.npz", ".json")
            datasetname = jsonfilename.split('_coh')[0]

        elif config['modelname'] == 'fragility':
            jsonfilename = f.name.replace('fragmodel.npz', 'frag.json')
            datasetname = jsonfilename.split('_frag.json')[0]

        elif config['modelname'] == 'impulse':
            jsonfilename = f.name.replace('.npz', '.json')
            datasetname = jsonfilename.split('_impulse')[0]

        elif config['modelname'] == 'correlation':
            # jsonfilename = f.name.replace('_corrmodel.npz', '')
            jsonfilename = f.name.replace("model.npz", ".json")
            datasetname = jsonfilename.split('_corr.json')[0]
        elif config['modelname'] == 'freq':
            jsonfilename = f.name.replace("model.npz", ".json")
            datasetname = jsonfilename.replace(".json", "")
        # elif config['modelname'] == 'coherence':
        #     jsonfilename = f.name.replace("model.npz", ".json")
        #     datasetname = jsonfilename.split('_coh.json')[0]
        else:
            jsonfilename = f.name.replace("model.npz", ".json")
            datasetname = jsonfilename.split("_svdmodel.json")[0]


        # if config['datatype'] == 'networkdegree':
        #     # jsonfilename = f.name.replace(".npz", ".json")
        #     # jsonfilename = jsonfilename.split("_degree")[0]
        #     datasetname = jsonfilename.split(".json")[0]
        # elif config['datatype'] == 'networksvd':
        #     # jsonfilename = jsonfilename.split("_svd")[0]
        #     datasetname = jsonfilename.split(".json")[0]
        # elif config['datatype'] == 'freq':
        #     # jsonfilename = jsonfilename.split("_freq")[0]
        #     datasetname = jsonfilename.split(".json")[0]
        # elif config['datatype'] == 'fragility_orthogonal':
        #     # jsonfilename = jsonfilename.split(".npz")[0] + ".json"
        #     datasetname = jsonfilename.split(".json")[0]

        if f.name.startswith("."):
            continue

        # get the actual dataset name
        # if 'umf004' in f.name:
        #     continue
        if 'jh106' in f.name:
            continue
        # if not 'pt17' in f.name:
        #     continue
        if ignore_interictal:
            if 'ii' in f.name:
                continue
        # if not any(x in f.name for x in ['nl01',
        #                                  'nl02', 'nl04', 'nl07', 'nl14', 'nl16', 'nl19'
        #                                  ]):
        #     continue
        # if not 'pt1' == f.name.split("_")[0]:
        #     continue
        # if not 'jh107' in f.name:# and not 'jh108' in f.name:
        #     continue

        # set the dictionary
        resultfiles_dict_json[jsonfilename] = datasetname
        # str(os.path.join(resultsdir, f.name))
    return resultfiles_dict_json


'''
Utility functions for preformatting a dataset from its raw .edf version.
'''


def rename_file(filepath):
    # gets rid of spaces -> '_'
    newfilepath = filepath.replace(' ', '_')

    # gets rid of & signs -> '-'
    newfilepath = newfilepath.replace("&", '-')

    os.rename(filepath, newfilepath)
    return newfilepath


def get_edf_filepath(hashmap, key):
    return hashmap[key]


def pair_name_patients(config):
    # hack inserted to make sure all modalities are for ieeg
    modality = config['modality']
    if modality == 'ieeg':
        modality = 'seeg'

    fif_path = Path(os.path.join(config['rawdatadir'],
                                 config['center'])).glob("*/{}/fif/*.fif".format(modality))
    # pair edf name to infile path using a dictionary
    fif_infiles_dict = {}
    for f in fif_path:
        # get the actual dataset name
        fname = f.name

        if fname.startswith('.'):
            continue

        # seeg datasets to ignore in this rule
        if modality == 'ieeg' or modality == 'seeg':
            if 'la07' in fname or 'la13' in fname:
                continue

        # create the new filepath we want the output of this pipeline to be
        fifdir = os.path.dirname(str(f))
        # use preformatter filename mapping - TRIMMING DATASET FILENAME
        fif_name = BaseMeta.map_filename(fname)

        patient_id = fif_name.split("_")[0]

        # set the dictionary
        fif_infiles_dict[str(os.path.join(fifdir, fif_name))] = patient_id

    return fif_infiles_dict


def pair_name_edffiles(config, ignore_interictal=False, ignore_ictal=False):
    # hack inserted to make sure all modalities are for ieeg
    modality = config['modality']
    # if modality == 'ieeg':
    #     modality = 'seeg'

    edf_path = Path(os.path.join(config['rawdatadir'],
                                 config['center'])).glob("*/{}/edf/*.edf".format(modality))

    if not edf_path:
        modality = 'ieeg'
        edf_path = Path(os.path.join(config['rawdatadir'],
                                 config['center'])).glob("*/{}/edf/*.edf".format(modality))
    # pair edf name to infile path using a dictionary
    edf_infiles_dict = {}
    for f in edf_path:
        # get the actual dataset name
        edf_name = f.name

        if edf_name.startswith('.'):
            continue

        # seeg datasets to ignore in this rule
        if modality == 'ieeg' or modality == 'seeg':
            if edf_name.startswith('.'):
                continue

        # create the new filepath we want the output of this pipeline to be
        fifdir = os.path.dirname(str(f)).replace('edf', 'fif')
        edfdir = fifdir.replace('fif', 'edf')

        edf_filepath = rename_file(os.path.join(edfdir, f.name))
        edf_name = os.path.basename(edf_filepath)

        # use preformatter filename mapping - TRIMMING DATASET FILENAME
        fif_name = BaseMeta.map_filename(edf_name)

        # if not any(x  in fif_name for x in [
            # 'nl01'
        #     # 'la21',
        #     # 'la22',
        #     # 'la23',
        #     # 'la24'
        #     # 'tvb'
        # ]):
        #     continue

        # extract root directory
        pat_id = os.path.basename(walk_up_folder(edf_filepath, 4))
        if pat_id not in fif_name:
            fif_name = pat_id + fif_name

        if modality == 'scalp':
            fif_name = fif_name.replace('_scalp', '')
        elif modality == 'seeg':
            fif_name = fif_name.replace("_seeg", "")

        if ignore_interictal:
            if 'ii' in fif_name:
                continue
        if ignore_ictal:
            if 'sz' in fif_name:
                continue

        # set the dictionary
        edf_infiles_dict[str(os.path.join(fifdir, fif_name))] = str(edf_filepath)

    return edf_infiles_dict
