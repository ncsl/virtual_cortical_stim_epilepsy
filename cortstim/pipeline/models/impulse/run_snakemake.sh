#!/bin/bash

ulimit -s 8196
#ulimit -s 8196
# CALL THIS FILE TO BEGIN SNAKEMAKE

# calls with cluster.json file options

#interact -p debug -n 4 -t 1:0:0

snakemake -j 4000 --cluster-config ../../config/cluster.json --cluster \
"sbatch --job-name={cluster.jobname} \
--mail-type=end \
--mail-user={cluster.account} \
--partition={cluster.partition} \
--nodes={cluster.nodes}  --time={cluster.time} \
--ntasks-per-node={cluster.ntasks-per-node} \
--cpus-per-task={cluster.cpus-per-task} \
--out={cluster.output} \
--error={cluster.error} \
--array={cluster.array}"

#--mem-per-cpu={cluster.mem}"

exit