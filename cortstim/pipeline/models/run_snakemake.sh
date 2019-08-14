#!/bin/bash
#SBATCH --job-name=FragilitySubmission
#SBATCH --time=24:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu

module restore conda
conda activate eztrack

rm -rf .snakemake/
rm -rf ./eztrack_out/
rm slurm*

# echo which python we are using
echo 'PYTHON IS $(which python)'

ulimit -s 8196
#ulimit -s 8196
# CALL THIS FILE TO BEGIN SNAKEMAKE

# calls with cluster.json file options

#interact -p debug -n 4 -t 1:0:0

# make sure job nubmer is greater then the total number of jobs you will submit!
#snakemake -j 2000 --cluster-config ../config/cluster.json --cluster \
#"sbatch --job-name={cluster.jobname} \
#--mail-type=end \
#--mail-user={cluster.account} \
#--partition={cluster.partition} \
#--nodes={cluster.nodes}  --time={cluster.time} \
#--ntasks-per-node={cluster.ntasks-per-node} \
#--cpus-per-task={cluster.cpus-per-task} \
#--out={cluster.output} \
#--error={cluster.error} \
#--exclusive;"

snakemake -j 500 --use-conda \
--cluster-config ../config/cluster.json \
--cluster \
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

#--mem-per-cpu={cluster.mem} \
exit