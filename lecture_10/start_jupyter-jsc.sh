#!/bin/bash
# this script (start_jupyter-jsc.sh) must be in the default $HOME/.jupyter to be used by Juypter-JSC

module purge
module use $OTHERSTAGES
module load Stages/Devel-2020
module load GCCcore/.9.3.0
module load JupyterCollection/2020.2.5
module load OpenAI-Gym/0.18.0-Python-3.8.5
module load PyTorch/1.7.0-Python-3.8.5
module load torchvision/0.8.1-Python-3.8.5
