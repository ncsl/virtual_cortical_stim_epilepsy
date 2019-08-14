import argparse
import os
import sys

import pandas as pd

sys.path.append('../../../')
from cortstim.edp.utils.utils import walk_up_folder, loadjsonfile, writejsonfile

parser = argparse.ArgumentParser(prog='Preformatting EDP data with the Python WM Contacts.', description='')
parser.add_argument('outputjson_filepath', type=str, help='Filepath for the desired outputfile!')
parser.add_argument('wmexceldatafilename', default="",
                    help='The filepath for the excel file to help merge and augment our metadata')


def get_out_contacts(elec_df):
    out_contacts = []

    # convert entire dataframe to upper case
    elec_df = elec_df.apply(lambda x: x.astype(str).str.upper())
    elec_df.iloc[0].apply(int)

    # loop over rows and search for 'WM'
    for idx, (index, row) in enumerate(elec_df.iterrows()):
        # get the contact numbers
        if idx == 0:
            contactnums = row.astype(int).tolist()

        for jdx, region in enumerate(row):
            if region == 'OUT':
                out_contacts.append(row.name + str(contactnums[jdx]))

    out_contacts = [x.lower() for x in out_contacts]
    return out_contacts


def get_wm_contacts(elec_df):
    wm_contacts = []

    # convert entire dataframe to upper case
    elec_df = elec_df.apply(lambda x: x.astype(str).str.upper())
    elec_df.iloc[0].apply(int)

    # loop over rows and search for 'WM'
    for idx, (index, row) in enumerate(elec_df.iterrows()):
        # get the contact numbers
        if idx == 0:
            contactnums = row.astype(int).tolist()

        # loop through each element in the row
        for jdx, region in enumerate(row):
            if region == 'WM':
                wm_contacts.append(row.name + str(contactnums[jdx]))
    wm_contacts = [x.lower() for x in wm_contacts]
    return wm_contacts


def loadmetadata(jsonfilepath):
    metadata = loadjsonfile(jsonfilepath)
    return metadata


def get_patientdir(rawdatasetpath):
    patdir = walk_up_folder(rawdatasetpath, 4)
    return patdir


if __name__ == '__main__':
    args = parser.parse_args()
    outputjson_filepath = args.outputjson_filepath
    excelfilename = args.wmexceldatafilename

    # get the excelfilepath
    patdatadir = get_patientdir(outputjson_filepath)
    excelfilepath = os.path.join(patdatadir, "seeg", excelfilename)

    # get the corresponding json file name that would be for the raw dataset
    outputjson_filepath = outputjson_filepath.replace('_raw.fif', '.json')

    # load in metadata and clinical df object
    metadata = loadmetadata(outputjson_filepath)
    eleclayout = pd.read_excel(excelfilepath, header=None, index_col=0, names=None)

    # get the contacts of interest
    wm_contacts = get_wm_contacts(eleclayout)
    out_contacts = get_out_contacts(eleclayout)

    # get wm contacts from metadata
    # extra_wm_contacts = []
    # lobe_mapping_contacts = metadata['ch_lobe_mapping']
    # for ch, lobeval in lobe_mapping_contacts.items():
    #     if lobeval == 'wm':
    #         extra_wm_contacts.append(ch)
    # wm_contacts.extend(extra_wm_contacts)

    # merge in data based on patient id and dataset id
    metadata['wm_contacts'] = [x.lower() for x in wm_contacts]
    metadata['out_contacts'] = [x.lower() for x in out_contacts]

    # write newly merged metadata
    writejsonfile(metadata, outputjson_filepath, overwrite=True)
