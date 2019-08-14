#!/usr/bin/env bash

center="clevelandnl"
modality="ieeg"
reference="common_avg"
modeltype="fragility"

resultsdatadir=/home/adam2392/hdd/data/processed/output_new/${modeltype}/${reference}/${modality}/${center}/
#resultsdatadir=/Users/adam2392/Downloads/output_new/${modeltype}/${reference}/${modality}/${center}/
#resultsdatadir=/Users/adam2392/Downloads/output_new/l2regularized/

marcc_scratchdatadir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/scratch/users/ali39@jhu.edu/data/processed/output_new/${modeltype}/${reference}/${modality}/${center}/
marcc_figsdir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/scratch/users/ali39@jhu.edu/data/figures
#figsdir=/Users/adam2392/Downloads/marccfigs/

# run rsync
#rsync --dry-run -aP $marcc_scratchdatadir $resultsdatadir;
rsync -aP $marcc_scratchdatadir $resultsdatadir;
#rsync -aP $marcc_figsdir $figsdir
