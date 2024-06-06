# rotarod-gait-analysis

## About 
This program uses a trained DeepLabCut model to perform pose estimation on recorded rotarod runs from a Parkinson's Disease mouse model [1]. The output data is then analyzed using NumPy and Pandas to return the scores for the gait signatures defined in Dela Cruz, et al. (2020). 

## Prepare Environment

Before the first use only, run `python setup.py` to update tracking folder home directory.

To create the conda environment: `conda env create -f DEEPLABCUT.yaml`

Download model weights from [this link](https://drive.google.com/drive/folders/1UmC0r9P78xBL41j9dniOAimbiRw42LPc?usp=sharing) into the folder `rotarod-gait-analysis/automated_analysis/dlc-models/iteration-2/Trial9May23-trainset95shuffle1/train`

## How to Use

Prepare a video folder containing the videos of interest and a .csv file containing the video filenames, start time (optional), and end time (optional). See `comparison_id.csv` template for an example (feel free to download). 

To run analysis, first activate the conda environment (`conda activate DLC_gait`) and change directory to `rotarod-gait-analysis`.

Next, run the following line of code:

```
python rotarod-gait-analysis/experiments/compare_runs.py --video_folder {video_folder_path} --output_folder {output_folder_path}
```

If the `output_folder` is not specified, the program will create a new output folder in the same root directory as the video folder.


## References

This work is used in the following paper:

Dela Cruz, H. L., Dela Cruz, E. L., Zurhellen, C. J., York, H. T., Baun, J. A., Dela Cruz, J. L., & Dela Cruz, J. S. (2020). New insights underlying the early events of dopaminergic dysfunction in Parkinson’s Disease. In *bioRxiv (Cold Spring Harbor Laboratory). Cold Spring Harbor Laboratory*. https://doi.org/10.1101/2020.09.27.313957
