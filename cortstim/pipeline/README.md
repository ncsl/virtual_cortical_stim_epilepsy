# Pipelines
Here, I describe the pipelines that are available to be run. 

TBD: A master pipeline file that links all pipelines together to run data analysis given EEG data, neuroimaging data and
data from the clinicians in a certain format. I still need to determine the optimal data structuring for this.

## 1. Preformat
This is the first pipeline to be run that converts .edf files into .fif files.

## 2. Neuroimaging
This is the pipeline that runs reconstruction, co-registration, and file parsing to get all usable files
for the user to get electrode coordinates, atlas parcellations and also surface geometries.

## 3. Models (Freq)
Runs frequency analysis of EEG data.

## 3. Models (Fragility)
Runs fragility analysis of EEG data.

## 4. Postformat
Converts all .npz + .json pairs of resulting data to .mat files for easy reading with MATLAB.

## 5. Classifiers
Runs classification analysis.


## 6. Deeplearning
Runs deep learning models training and testing pipelines.
