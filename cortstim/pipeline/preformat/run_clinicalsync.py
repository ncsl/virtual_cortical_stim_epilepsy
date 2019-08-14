# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('../../../')
from cortstim.edp.utils.utils import walk_up_folder, loadjsonfile, writejsonfile
import time
import argparse

from cortstim.edp.objects.clinical.clinical_object import ClinicalMeta
from cortstim.edp.loaders.dataset.clinical.excel_meta import ExcelReader

parser = argparse.ArgumentParser(prog='Preformatting EDP data.', description='')
parser.add_argument('rawdatadir', type=str, help="Directory of data to analyze")
parser.add_argument('outputjson_filepath', type=str, help='Filepath for the desired outputfile!')
parser.add_argument('datatype', default=None, help="Specific datatype to preformat (seeg, scalp, ecog)")
parser.add_argument('exceldatafilepath', default="",
                    help='The filepath for the excel file to help merge and augment our metadata')


def loadmetadata(jsonfilepath):
    metadata = loadjsonfile(jsonfilepath)
    return metadata


def loadclinicaldf(excelfilepath):
    clinreader = ExcelReader(excelfilepath)
    clinreader.read_formatted_df(excelfilepath)
    return clinreader


def merge_metadata(metadata, clinreader, pat_id, dataset_id, modality='ieeg'):
    # get the clinical for this patient
    patclindf = clinreader.ieegdf

    # get the clinical data for this dataset
    datasetdf = clinreader.datasetdf

    scalpdf = clinreader.scalpdf

    # trim the dataframes
    patclindf._trimdf(patient_id=pat_id)
    datasetdf._trimdf(patient_id=pat_id)
    datasetdf.trimdataset(dataset_id=dataset_id)
    scalpdf._trimdf(patient_id=pat_id)

    # load in the clinical metadatafile
    clinicalmeta = ClinicalMeta(metadata)
    print("\nAugmented patient specific df")
    # merge those into the metadata
    clinicalmeta.augment_metadata(patclindf.clindf)

    print(datasetdf.clindf)
    # augment per dataset or by scalp eeg
    if modality == 'scalp':
        print("\nAugmented scalp specific df")
        clinicalmeta.augment_metadata(scalpdf.clindf)
    elif modality == 'ieeg':
        print("\nAugmented dataset df")
        clinicalmeta.augment_metadata(datasetdf.clindf)

    metadata = clinicalmeta.metadata

    return metadata


if __name__ == '__main__':
    args = parser.parse_args()
    rawdatadir = args.rawdatadir
    outputjson_filepath = args.outputjson_filepath
    modality = args.datatype
    excelfilepath = args.exceldatafilepath

    # time the program
    start = time.time()

    outputjson_filepath = outputjson_filepath.replace('_raw.fif', '.json')

    print("saving to ", outputjson_filepath)
    ''' EXTRACT USEFUL INFORMATION FROM THE OUTPUTJSON FILEPATH '''
    # extract the clinical center from the path name
    clinical_center = os.path.basename(os.path.normpath(rawdatadir))

    # extract root directory
    pat_id = os.path.basename(walk_up_folder(outputjson_filepath, 4))
    dataset_id = ''.join(os.path.basename(outputjson_filepath.replace('_raw', '')).split('.')[0].split('_')[1:])

    # load in metadata and clinical df object
    metadata = loadmetadata(outputjson_filepath)
    clindf = loadclinicaldf(excelfilepath)

    if 'la' in pat_id:
        dataset_id = dataset_id.replace("scalp", '').replace("sz", '')

    print("looking at: ", pat_id, dataset_id)

    # merge in data based on patient id and dataset id
    metadata = merge_metadata(metadata, clindf, pat_id, dataset_id, modality=modality)

    # write newly merged metadata
    writejsonfile(metadata, outputjson_filepath, overwrite=True)
