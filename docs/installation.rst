Installation
============

Prerequisites
~~~~~~~~~~~~~~

You can currently install liam with Conda.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.8.

liam installation
~~~~~~~~~~~~~~~~~~~~~~~
# create an anaconda environment from the provided yml file

$ conda env create -f liam_dependencies.yml


# activate the created environment

$ conda activate liam_env


# install liam

$pip install https://github.com/ohlerlab/liam/archive/refs/tags/v1.0.0.zip


Liam was tested and developed in Python 3.8 on a Linux system running CentOS 7 using a Tesla-T4 graphic or Tesla-V100-SXM2-32GB graphic card with CUDA 11.3.