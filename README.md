# rotarod-gait-analysis

## About 
This program uses a trained DeepLabCut model to perform pose estimation on recorded rotarod runs from a Parkinson's Disease mouse model [1]. The output data is then analyzed using NumPy and Pandas to return the scores for the gait signatures defined in Dela Cruz, et al. (2020). To use this program please prepare a DeepLabCut folder as per [DeepLabCut's Official Documentation](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#deeplabcut-in-the-terminal-command-line-interface) for running the initial pose estimation. In addition, include a `.csv` file containing the corresponding filenames to be compared for gait signature analysis (see template).

## References

This work is used in the following paper:

Dela Cruz, H. L., Dela Cruz, E. L., Zurhellen, C. J., York, H. T., Baun, J. A., Dela Cruz, J. L., & Dela Cruz, J. S. (2020). New insights underlying the early events of dopaminergic dysfunction in Parkinson’s Disease. In *bioRxiv (Cold Spring Harbor Laboratory). Cold Spring Harbor Laboratory*. https://doi.org/10.1101/2020.09.27.313957
