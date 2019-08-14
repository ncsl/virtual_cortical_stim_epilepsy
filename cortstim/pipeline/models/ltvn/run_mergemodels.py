import argparse
import os
import sys

sys.path.append("../../../")
sys.path.append("../../../../")

from cortstim.edm.fragility.execute.runmerge import RunMergeModel
from cortstim.edp.utils.utils import walk_up_folder


def main(outputfilename, outputmetafilename, tempdir):
    tempfiles = [f for f in os.listdir(tempdir) if not f.startswith('.')]
    numwins = len(tempfiles)

    if not os.path.exists(outputmetafilename):
        temp0file = os.path.join(tempdir, "temp0.npz")
        os.remove(temp0file)
        # remove success file
        outfile = os.path.basename(outputfilename)
        success_flag_name = outfile.replace('fragmodel.npz', 'frag_success.txt')
        success_flag_name = os.path.join(walk_up_folder(tempdir, 2), success_flag_name)
        os.remove(success_flag_name)
        sys.stdout.write("File Removed {}!".format(success_flag_name))
        raise RuntimeError("Number of windows is not the same as the number of"
                           "files in the temp dir. Should be {}".format(numwins))

    # run merge
    merger = RunMergeModel(tempdir, numwins)
    merger.loadmetafile(outputmetafilename)
    metadata = merger.metadata
    if not metadata['numwins'] == numwins:
        # remove success file
        outfile = os.path.basename(outputfilename)
        success_flag_name = outfile.replace('fragmodel.npz', 'frag_success.txt')
        success_flag_name = os.path.join(walk_up_folder(tempdir, 2), success_flag_name)
        os.remove(success_flag_name)
        sys.stdout.write("File Removed {}!".format(success_flag_name))

        raise RuntimeError("Number of windows is not the same as the number of"
                           "files in the temp dir. Should be {} vs #files: {}".format(metadata['numwins'], numwins))

    merger.loadmetadata(metadata)
    merger.mergefragilitydata(outputfilename=outputfilename)
    merger.writemetadata_tofile(outputmetafilename=outputmetafilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outputfilename', help="The output model computed filename.")
    parser.add_argument('outputmetafilename', help="The output meta data filename")
    parser.add_argument('tempdir', help="The temporary directory to save per window compute.")

    args = parser.parse_args()

    outputfilename = args.outputfilename
    outputmetafilename = args.outputmetafilename
    tempdir = args.tempdir

    # analyze the data
    main(outputfilename, outputmetafilename, tempdir)
