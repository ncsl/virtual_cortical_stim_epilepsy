# Virtual Cortical Stimulation For Seizure Localization
[![Build Status](https://travis-ci.com/adam2392/cortstimv2.svg?token=6sshyCajdyLy6EhT8YAq&branch=master)](https://travis-ci.com/adam2392/cortstimv2)
[![Coverage Status](./coverage.svg)](./coverage.svg)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Documentation Status]()]()

A repository for running virtual cortical stimulation on linear time-varying network based models. Paper is here: 

Maintained(ish) by: Adam Li (adam2392 at gmail dot com)

# Installation and Set Up
1. conda and conda-env


    conda create -n cortstim
    source activate cortstim
    conda env create -f environment.yaml python=3.6
    conda install --file requirements_conda.txt

or

    conda create -n cortstim scipy numpy pandas scikit-learn matplotlib seaborn pyqt=5 ipykernel pytest
    conda install mayavi   
    conda install -c conda-forge mne
    conda install -c anaconda natsort pylint flake8 tqdm
    conda install -c bioconda snakemake
    conda install -c numba numba
    conda install xlrd
        
# Modules
Here, we describe the main modules to run the entire software package.

1. Epilepsy Data Platform (EDP)
The main i/o software for dealing with data that are used in this software. 
We generally support edf, fif, and any related MNE-python supported data format. 
We provide a preformatter pipeline to convert some raw data into our desired .fif/.json format.

2. Epilepsy Data Models (EDM)
The main compute package that supports running on a single core, multiple core, or SLURM HPC environment. 
It will run the complete algorithm from data input read in with the IO module.

# Testing

    autopep8 --in-place --recursive --max-line-length=80 ./cortstim/
    autopep8 --in-place --recursive --max-line-length=80 ./tests/
    pylint ./cortstim/
    pylint ./tests/
    pytest --cov-config=.coveragerc --cov=./cortstim/ tests/
    coverage-badge -f -o coverage.svg

# Documentation
If you rerun documentation and want to generate it into sphinx-docs.

    sphinx-quickstart
    
# House Keeping
Some useful commands for housekeeping on conda.

    conda clean --all
    conda update --all
    conda update conda
    conda env export > environment.yaml
    
# Ipython Kernel Updating
Adding a kernel to jupyter from your conda environment

    conda create -n cortstim scipy numpy pandas scikit-learn matplotlib seaborn pyqt=5 ipykernel pytest
    conda install mayavi   
    conda install -c conda-forge mne
    conda install -c anaconda natsort pylint flake8 tqdm
    conda install -c bioconda snakemake
    conda install -c numba numba
    conda install xlrd
    python -m ipykernel install --name cortstim --user 
    jupyter kernelspec uninstall yourKernel
    jupyter kernelspec list
    

# Data Directory Structuring
All data analysis assumes a structure:

- higher level (e.g. center, organized batch of datasets)
    - patient_id
        - seeg
            - edf
            - fif
        - scalp
            - edf
            - fif
            
# References:
1. https://ieeexplore.ieee.org/document/7963378
2. https://ieeexplore.ieee.org/document/8037439



