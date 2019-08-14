#!/bin/bash
#SBATCH --job-name=FragilitySubmission
#SBATCH --time=24:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu

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

exit