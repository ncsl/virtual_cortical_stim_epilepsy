import argparse
import os
import sys

from natsort import natsorted

sys.path.append("../../../")
sys.path.append("../../../../")

from eztrack.edp.utils.utils import loadjsonfile


def walk_up_folder(path, depth=1):
    _cur_depth = 1
    while _cur_depth < depth:
        path = os.path.dirname(path)
        _cur_depth += 1
    return path


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def check_success_job(numwins, tempdatadir, output_filepath):
    success = True
    all_files = natsorted([f for f in os.listdir(tempdatadir)
                           if not f.startswith('.')])

    # perform checks on the computed data
    if len(all_files) != numwins:
        success = False
    else:
        # save adjacency matrix
        for idx, tempfile in enumerate(all_files):
            # get the window numbr of this file just to check
            buff = tempfile.split('_')
            winnum = int(buff[-1].split('.')[0])
            if winnum != idx:
                success = False
                break

    if success:
        touch(output_filepath)
    else:
        raise RuntimeError("There was an error when checking the success of this {}"
                           "file run. There were {} files, but supposed to be {}.".format(output_filepath,
                                                                                          len(all_files), numwins))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfilepath', help="File path to the json,fif pair dataset")
    parser.add_argument('output_success_file', help="The file to touch if success.")
    parser.add_argument('tempdir', help="The temporary directory to save per window compute.")

    args = parser.parse_args()

    # extract arguments from parser
    output_success_file = args.output_success_file
    tempdir = args.tempdir
    jsonfilepath = args.jsonfilepath

    metadata = loadjsonfile(jsonfilepath)
    numwins = metadata['numwins']

    sys.stdout.write(output_success_file)
    sys.stdout.write(tempdir)
    sys.stdout.write(jsonfilepath)
    sys.stdout.write("check success of job.")

    # check success
    check_success_job(numwins, tempdir, output_success_file)
