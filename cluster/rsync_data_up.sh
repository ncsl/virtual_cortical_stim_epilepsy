#!/usr/bin/env bash

center="umf"

rawdatadir=/home/adam2392/hdd2/data/rawdata/${center}/
#rawdatadir=/Users/adam2392/Downloads/tngpipeline/${center}/

marccdatadir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/home-1/ali39@jhu.edu/data/epilepsy_raw/${center}/
marccscratchdatadir=ali39@jhu.edu@gateway2.marcc.jhu.edu:/scratch/users/ali39@jhu.edu/data/epilepsy_raw/${center}/

# run rsync
rsync -aP $rawdatadir $marccdatadir;
rsync -aP $rawdatadir $marccscratchdatadir;