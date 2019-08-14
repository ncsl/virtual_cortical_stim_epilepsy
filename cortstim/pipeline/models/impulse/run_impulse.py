import argparse
import os
import sys

import numpy as np

sys.path.append("../../../")
sys.path.append("../../../../")

from cortstim.edp.utils.utils import writejsonfile
from cortstim.edm.fragility.model.impulsemodel import ImpulseModel

from cortstim.pipeline.utils.utils import load_ltvn_results


def main(ltvmat, metadata, outputfilepath, outputjsonfilepath):
    model_params = {
        "stabilize": True,
        "magnitude": 1,
    }

    print("Starting impulse response modeling", flush=True)

    impulsemodel = ImpulseModel(show_progress=False)
    l2norm_responses = impulsemodel.run(ltvmat, **model_params)

    print("Finished impulse response modeling", flush=True)
    # update metadata terms
    metadata['stabilize'] = model_params['stabilize']
    metadata['resultfilename'] = os.path.basename(outputfilepath)
    metadata['impulse_length'] = impulsemodel.N
    metadata['impulse_metric'] = 'l2norm'
    metadata['impulse_magnitude'] = model_params['magnitude']

    # save out the metadata
    writejsonfile(metadata, outputjsonfilepath, overwrite=True)
    np.savez_compressed(outputfilepath,
                        impulse_results=l2norm_responses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help="The result directory of the dataset")
    parser.add_argument('resultmetafilepath', help="The meta filepath for the ltvn result we want to analyze.")
    parser.add_argument('outputfilepath', help="The output model computed filename.")
    parser.add_argument('outputmetafilepath', help="The output meta data filename")

    args = parser.parse_args()

    # extract arguments from parser
    results_dir = args.results_dir
    resultmetafilepath = args.resultmetafilepath
    outputfilepath = args.outputfilepath
    outputmetafilepath = args.outputmetafilepath

    # debug print statements
    print("Running impulse model!", flush=True)
    print("Getting ltvn model results from: ", results_dir, flush=True)

    # load in the data
    ltvmodel = load_ltvn_results(results_dir, resultmetafilepath)
    ltvmat = ltvmodel.get_data()
    ltvmat = np.rollaxis(ltvmat, 2, 0)
    metadata = ltvmodel.get_metadata()

    print("ltvmat before: ", ltvmat.shape)
    try:
        onsetwin = ltvmodel.onsetwin
        ltvmat = ltvmat[onsetwin - 30:, ...]
        metadata['onsetwin'] = 30
    except:
        print("Not trimming")
        raise Exception("Onsetwin can't be set for this dataset. Rerun.")
    try:
        offsetwin = ltvmodel.offsetwin
        print("onset and offsetwin here are: ", onsetwin, offsetwin)
        ltvmat = ltvmat[:offsetwin + 30, ...]
        metadata['offsetwin'] = ltvmat.shape[1] - 30
    except:
        print("Not trimming")

    try:
        print(metadata['ez_hypo_contacts'], metadata['seizure_semiology'], metadata['ablated_contacts'], flush=True)
    except:
        print(metadata.keys(), flush=True)
    print("Shape of the ltvn model:", ltvmat.shape, flush=True)

    # analyze the data
    main(ltvmat, metadata, outputfilepath, outputmetafilepath)
