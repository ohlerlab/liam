# Liam

Liam (**l**everaging **i**nformation **a**cross **m**odalities) is an adversarial variational autoencoder-based model for the simultaneous
vertical (derives a joint low-dimensional embedding informed by both modalities) and horizontal (batch) integration of paired multimodal
scRNA-seq and scATAC-seq data, and scRNA-seq and ADT data (CITE-seq; Liam_ADT). It can also integrate paired with unimodal data sets (mosaic integration).

If you are using liam, please cite [[1]](#1).

For legacy software used for analyses presented in [[1]](#1), see: https://github.com/ohlerlab/liam_challenge_reproducibility

For analysis scripts see: https://github.com/ohlerlab/liam_manuscript_reproducibility

# Installation
You can install liam with [anaconda](https://www.anaconda.com/) following the listed steps:

```
# create an anaconda environment from the provided yml file
conda env create -f liam_dependencies.yml

# activate the created environment
conda activate liam_env

# install liam  
pip install https://github.com/ohlerlab/liam/archive/refs/tags/v1.0.0.zip
```
Liam was tested and developed in Python 3.8 on a Linux system running CentOS 7 using a Tesla-T4 graphic or Tesla-V100-SXM2-32GB graphic card with CUDA 11.3.

# Usage example
For a tutorial showing how to use liam for paired multimodal data integration, see [Tutorial paired](tutorials/notebooks/Liam_usage_example.ipynb); for a tutorial showing how to use liam in a mosaic integration scenario that includes scaling of the adversarial loss, see [Tutorial mosaic](tutorials/notebooks/Liam_usage_example_mosaic_adversary_x5.ipynb).

# Third-Party Acknowledgements
Liam was built using the scvi-tools-skeleton repository (version 0.4.0) as a template and contains modified code from scvi-tools (version 0.14.3) [[2]](#2).
Additionally, liam adopts a negative binomial loss function for the reconstruction loss of the scATAC-seq data recently introduced by [[3]](#3).
To this end, I translated the source code of the negative binomial loss function from [BAVARIA](https://github.com/BIMSBbioinfo/bavaria) [[3]](#3) to the PyTorch framework, which I indicate in the docstrings of the respective functions.
The author of BAVARIA, Wolfgang Kopp, granted an exception to use this loss function under a BSD 3-Clause License. Lastly, liam implements an experimental feature - batch adversarial training, as introduced by [[4]](#4) and independently implemented by [[3]](#3). In liam, I use an implementation that uses a gradient reversal layer as proposed in [[4]](#4). The gradient reversal layer used in liam stems from a public GitHub repository: fungtion/DANN.

The licenses of the third-party code are linked in the following and can be found in the folder [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES):
- scvi-tools-skeleton: [license](THIRD-PARTY-LICENSES/scvi-tools-skeleton-LICENSE)
- scvi-tools: [license](THIRD-PARTY-LICENSES/scvi-tools-LICENSE)
- DANN: [license](THIRD-PARTY-LICENSES/DANN-LICENSE)

## References
<a id="1">[1]</a>
Rautenstrauch, P. & Ohler, U. (2022) [Liam tackles complex multimodal single-cell data integration challenges](https://www.biorxiv.org/content/10.1101/2022.12.21.521399v1). bioRxiv DOI: 10.1101/2022.12.21.521399

<a id="2">[2]</a>
Gayoso, A. et al. (2022) A Python library for probabilistic analysis of single-cell omics data. Nature Biotechnology DOI: 10.1038/s41587-021-01206-w

<a id="3">[3]</a>
Kopp, W. et al. (2022) Simultaneous dimensionality reduction and integration for single-cell ATAC-seq data using deep learning. Nature Machine Intelligence DOI: 10.1038/s42256-022-00443-1

<a id="4">[4]</a>
Ganin, Y. et al. (2016) Domain-Adversarial Training of Neural Networks. J. Mach. Learn. Res. 17, 1–35
