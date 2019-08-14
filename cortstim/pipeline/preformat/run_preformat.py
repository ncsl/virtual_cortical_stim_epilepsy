import os
import sys

sys.path.append('../../../')
from cortstim.edp.format.formatter_raw import ConvertEDFiEEG, ConvertEDFScalp
from cortstim.edp.utils.utils import walk_up_folder
import time
import argparse

parser = argparse.ArgumentParser(prog='Preformatting EDP data.', description='')
parser.add_argument('rawdatadir', type=str, help="Directory of data to analyze")
parser.add_argument('inputedf_filepath', type=str, help='Filepath to input edf file')
parser.add_argument('outputjson_filepath', type=str, help='Filepath for the desired outputfile!')
parser.add_argument('datatype', default=None, help="Specific datatype to preformat (seeg, scalp, ecog)")


def find_corresponding_edf_file(path, filename):
    """
    Utility function
    :param path:
    :param filename:
    :return:
    """
    dataset_paths = [x for x in os.listdir(path) if x.replace(' ', '_').lower() == filename
                     if x.endswith('.edf') if
                     not x.startswith('.')]
    return dataset_paths[0]


def find_underscores(edffilename):
    """
    Utility function
    :param edffilename:
    :return:
    """
    str_copy = list(edffilename).copy()
    indices = []
    ind = len(str_copy) - 1

    # find indices of the '_'
    while len(str_copy) > 0:
        char = str_copy.pop()
        if char == '_':
            indices.append(ind)
        ind -= 1

    if len(indices) > 1:
        return indices[0]
    else:
        return None


def replace_str_index(text, index=0, replacement=''):
    """
    Utility function
    :param text:
    :param index:
    :param replacement:
    :return:
    """
    return '%s%s%s' % (text[:index], replacement, text[index + 1:])


def run_conversion(edffilepath, pat_id, dataset_id, json_file, datatype, clinical_center, outputfilepath, save=True):
    # initialize converter
    if datatype == 'seeg' or datatype == 'ieeg' or datatype == 'ecog':
        # initialize converter
        edfconverter = ConvertEDFiEEG(datatype=datatype)
    else:
        edfconverter = ConvertEDFScalp(datatype=datatype)

    print("LOOKING AT: ", edffilepath)

    # load in the dataset and create metadata object
    edfconverter.load_file(filepath=edffilepath)

    # load in info data structure and edf annotated events
    edfconverter.extract_info_and_events(json_events_file=json_file, pat_id=pat_id)

    rawfif = edfconverter.convert_fif(bad_chans_list=[], newfilepath=outputfilepath, save=save, replace=True)
    newjsonpath = os.path.join(fifdir, outputfilepath.replace('_raw.fif', '.json'))
    metadata = edfconverter.convert_metadata(pat_id, dataset_id, clinical_center, save=True, jsonfilepath=newjsonpath)

    # declare where to save these files
    # newfifname = metadata['filename']
    # newfifname = newfifname.replace('_scalp', '')
    # newfifpath = os.path.join(fifdir, newfifname)
    # newjsonpath = os.path.join(fifdir, newfifpath.replace('_raw.fif', '.json'))
    print(outputfilepath)
    print(newjsonpath)

    end = time.time()
    print("Done! Time elapsed: {:.2f} secs".format(end - start))

    return rawfif, metadata


if __name__ == '__main__':
    args = parser.parse_args()
    rawdatadir = args.rawdatadir
    inputedf_filepath = args.inputedf_filepath
    outputjson_filepath = args.outputjson_filepath
    datatype = args.datatype

    # time the program
    start = time.time()

    ''' EXTRACT USEFUL INFORMATION FROM THE OUTPUTJSON FILEPATH '''
    # extract the clinical center from the path name
    clinical_center = os.path.basename(os.path.normpath(rawdatadir))

    # extract root directory
    pat_id = os.path.basename(walk_up_folder(outputjson_filepath, 4))
    dataset_id = '_'.join(os.path.basename(outputjson_filepath.replace('_raw', '')).split('.')[0].split('_')[1:])
    patdatadir = os.path.join(rawdatadir, pat_id)

    # extract the actual json filename
    outputjson_filename = os.path.basename(outputjson_filepath)

    # set the directories according to how we decided
    seegdir = os.path.join(walk_up_folder(outputjson_filepath, 3), 'edf')
    fifdir = seegdir.replace('edf', 'fif')
    if not os.path.exists(fifdir):
        os.makedirs(fifdir)

    print("seegdir ", seegdir)

    # get to the corresponding edf file that will need to be converted
    edffilepath = inputedf_filepath
    print("looking at specific patid: ", pat_id, dataset_id)
    # print(json_file)
    print("Seegdir: ", seegdir)
    print("edffilepath: ", edffilepath)

    # set this if you want to import a custom json_file with metadata inside
    # json_file = inputedf_filepath.replace('.edf', '.json')
    json_file = None

    rawfif, metadata = run_conversion(edffilepath, pat_id, dataset_id, json_file, datatype, clinical_center,
                                      outputjson_filepath)
