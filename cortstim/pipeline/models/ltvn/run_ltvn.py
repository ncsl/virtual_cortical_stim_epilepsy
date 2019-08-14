import argparse
import os
import sys
import timeit

import numpy as np

sys.path.append("../../../")
sys.path.append("../../../../")

from cortstim.edm.fragility.execute.runltv import RunLTVModel
from cortstim.edp.utils.utils import walk_up_folder

from cortstim.pipeline.utils.utils import load_raw_data


def main_array(rawdata, metadata, outputfilename, outputmetafilename, tempdir,
               ARRAY_INDEX, ARRAY_SIZE, compute_missing_wins=True):
    # set the model paramters
    winsize = 250
    stepsize = 125
    samplerate = metadata['samplerate']
    stabilizeflag = False

    # run model
    model_params = {
        "winsize": winsize,
        "stepsize": stepsize,
        "samplerate": samplerate,
        "l2penalty": 1e-5,
        "stabilizeflag": stabilizeflag,
    }
    model = RunLTVModel(**model_params)

    model.load_data(rawdata)
    model.settempdir(tempdir)
    numwins = model.numwins
    samplepoints = model.samplepoints

    # partition sample windows based on size of array
    if compute_missing_wins:
        # print("Computing missing windows!", flush=True)
        missingwins = model.getmissingwins(tempdir, numwins)
        # split missing windows among array size
        winlist_split = np.array_split(missingwins, ARRAY_SIZE)
    else:
        winlist_split = np.array_split(range(0, numwins), ARRAY_SIZE)

    winlist = winlist_split[ARRAY_INDEX]
    if winlist == []:
        return 1

    print("channels: ", metadata['chanlabels'])
    # print("looking at these windows: {}".format(winlist), flush=True)
    # print("Total number of windows: {} vs {} vs {}".format(numwins, len(winlist), len(missingwins)), flush=True)

    update_kws = {
        **model_params,
        'resultfilename': os.path.basename(outputfilename),
        "samplepoints": samplepoints,
        "numwins": numwins
    }
    if not os.path.exists(os.path.dirname(outputmetafilename)):
        os.makedirs(os.path.dirname(outputmetafilename))

    if not os.path.exists(outputmetafilename):
        # save the existing model's metadata
        model.writemetadata_tofile(metadata, outputmetafilename, update_kws)
        # print("Saved metadata", flush=True)

    # run model over these windows
    for iwin in winlist:
        # print("Analyzing iwin: {}".format(iwin), flush=True)
        adjmat = model.runwindow(iwin=iwin)
        # print("Finished Analyzing iwin: {}".format(iwin), flush=True)


def main_mp(rawdata, metadata, outputfilename, outputmetafilename, tempdir, compute_missing_wins=True):
    # set the model paramters
    if metadata['modality'] == 'ieeg':
        winsize = 250
        stepsize = 125
    elif metadata['modality'] == 'scalp':
        winsize = 250
        stepsize = 125

    samplerate = metadata['samplerate']
    stabilizeflag = False

    # run model
    model_params = {
        "winsize": winsize,
        "stepsize": stepsize,
        "samplerate": samplerate,
        "stabilizeflag": stabilizeflag,
        "numcores": 4,
    }
    model = RunLTVModel(**model_params)

    model.load_data(rawdata)
    model.settempdir(tempdir)
    numwins = model.numwins
    samplepoints = model.samplepoints

    # start timer
    start_time = timeit.default_timer()

    # partition sample windows based on size of array
    if compute_missing_wins:
        print("Computing missing wins!")
        winlist = model.getmissingwins(tempdir, numwins)
    else:
        winlist = np.arange(0, numwins)

    print("channels: ", metadata['chanlabels'])
    print("Total number of windows: {} vs {}".format(numwins, len(winlist)), flush=True)

    # run model
    model.runmpall(compute_on_missing_wins=compute_missing_wins)
    # for iwin in winlist:
    #     model.runwindow(iwin)

    # end timer
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print("Elapsed time: {}".format(elapsed))

    update_kws = {
        "winsize": winsize,
        "stepsize": stepsize,
        "samplerate": samplerate,
        "stabilizeflag": stabilizeflag,
        'resultfilename': os.path.basename(outputfilename),
        "samplepoints": samplepoints,
        "numwins": numwins,
    }

    if not os.path.exists(os.path.dirname(outputmetafilename)):
        os.makedirs(os.path.dirname(outputmetafilename))

    # save the existing model's metadata
    model.writemetadata_tofile(metadata, outputmetafilename, update_kws)


# convert bipolar mapping to monopolar mapping
def bipolar_to_monopolar(bipchs):
    import re
    contact_single_regex = re.compile("^([A-Za-z]+[']?)([0-9]+)$")
    contact_pair_regex_1 = re.compile("^([A-Za-z]+[']?)([0-9]+)-([0-9]+)$")
    contact_pair_regex_2 = re.compile("^([A-Za-z]+[']?)([0-9]+)-([A-Za-z]+[']?)([0-9]+)$")

    monlabels = []

    # get the electrode label name
    elecnames = {}
    for name in bipchs:
        elecind = re.search("\d", name).start()
        elecname = name[:elecind]
        elecnumind = re.search("-", name).start()
        elecnum1 = name[elecnumind - 1]
        elecnum2 = name[elecnumind + 1]

        # match = contact_pair_regex_1.match(name)
        # elecname, elecnum1, elecnum2 = match.groups()
        elecnames[name] = [elecname + elecnum1, elecname + elecnum2]

        monlabels.extend([elecname + elecnum1, elecname + elecnum2])
    return monlabels


if __name__ == '__main__':
    print("Inside Fragility Analysis...", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help="The root directory of the dataset")
    parser.add_argument('jsonfilepath', help="File path to the json,fif pair dataset")
    parser.add_argument('outputfilename', help="The output model computed filename.")
    parser.add_argument('outputmetafilename', help="The output meta data filename")
    parser.add_argument('tempdir', help="The temporary directory to save per window compute.")
    parser.add_argument('modality', help="The Modality of the data to be run")
    parser.add_argument('reference', help="The reference scheme to apply to the data.")
    parser.add_argument('--gnuind', default=None, help="The index on gnu parallel job.")
    parser.add_argument('--gnusize', default=None, help="The size of the gnu parallel loop.")
    args = parser.parse_args()

    # extract arguments from parser
    root_dir = args.root_dir
    outputfilename = args.outputfilename
    outputmetafilename = args.outputmetafilename
    tempdir = args.tempdir
    jsonfilepath = args.jsonfilepath
    reference = args.reference
    modality = args.modality

    gnuind = args.gnuind
    gnusize = args.gnusize

    print("\n\nArguments are: {}".format(args), flush=True)
    print(root_dir, modality, flush=True)
    print(jsonfilepath, flush=True)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in [f for f in filenames if f == jsonfilepath]:
            jsonfilepath = os.path.join(dirpath, filename)

    # reassign root dir
    root_dir = walk_up_folder(jsonfilepath, 4)
    modality = os.path.basename(walk_up_folder(jsonfilepath, 3))

    root_dir = os.path.join(root_dir, modality, "fif")
    print("\nInside run frag...\n", flush=True)
    print(root_dir, modality, flush=True)
    print(jsonfilepath, flush=True)

    # load in the data
    eegts = load_raw_data(root_dir,
                          jsonfilepath,
                          reference,
                          apply_mask=True,
                          remove_wm_contacts=True,
                          modality=modality)

    rawdata = eegts.get_data()
    metadata = eegts.get_metadata()
    metadata['chanlabels'] = eegts.chanlabels

    # decimate signal in time
    # eegtschanlabels = np.array([ch.lower() for ch in eegts.chanlabels])
    # # eegtschanlabels = np.array([ch.split("m")[-1].lower() for ch in eegtschanlabels])
    #
    # bipchlabels = metadata['label_bipolar']
    # monchinds = np.unique(np.array(metadata['BipChOrder']).flatten()) - 1
    # print(monchinds)
    # chlabels = np.unique(bipolar_to_monopolar(bipchlabels))
    # chlabels = [ch.lower() for ch in chlabels]
    # # only keep these chlabels
    # # tokeepinds = [i for i, ch in enumerate(eegtschanlabels) if ch in chlabels]
    # tokeepinds = monchinds.astype(int)
    # chlabels = eegtschanlabels[tokeepinds]
    # rawdata = rawdata[tokeepinds,:]
    # metadata['chanlabels'] = chlabels
    #
    # print("Normal contacts", eegtschanlabels)
    # print("Keeping contacts", chlabels)
    # print("Stripping contacts")
    # print(rawdata.shape, len(chlabels))
    # assert len(chlabels) == rawdata.shape[0]

    try:
        if gnusize is not None and gnuind is not None:
            print("Got type error in trying slurm array", flush=True)
            ARRAY_TASK_COUNT = int(gnusize)
            ARRAY_INDEX = int(gnuind)

            main_array(rawdata, metadata, outputfilename, outputmetafilename, tempdir,
                       ARRAY_INDEX,
                       ARRAY_TASK_COUNT)
        else:
            # extract slurm job parametesr
            SLURM_ARRAY_TASK_COUNT = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))

            # index starts at 0
            SLURM_ARRAY_INDEX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

            main_array(rawdata, metadata, outputfilename, outputmetafilename, tempdir,
                       SLURM_ARRAY_INDEX,
                       SLURM_ARRAY_TASK_COUNT)
    except:
        print("Got an error.")
        print("Got exception when trying gnu parallel: {}", flush=True)
        print("Running locally cause can't read slurm vars!", flush=True)
        # analyze the data
        main_mp(rawdata, metadata, outputfilename, outputmetafilename, tempdir)
        # raise Exception("Messed up in run_impulse.py! {}".format(e))
