#!/usr/bin/env bash

center="cleveland"
modality="ieeg"
reference="bipolar"
modeltype="fragility"


#resultsdatadir="/home/adam2392/hdd/data/output_new/${center}/"
resultsdatadir=/Users/adam2392/Downloads/output_new/${modeltype}/${reference}/${modality}/${center}/

marcc_scratchdatadir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/scratch/users/ali39@jhu.edu/data/processed/output_new/${modeltype}/${reference}/${modality}/${center}/
marcc_figsdir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/scratch/users/ali39@jhu.edu/data/figures
figsdir=/Users/adam2392/Downloads/marccfigs/

# run rsync
#rsync --dry-run -aP $marcc_scratchdatadir $resultsdatadir;
rsync -aP $resultsdatadir $marcc_scratchdatadir;