#!/bin/bash

set -euxo pipefail

# export CUDA_HOME=/usr/local/cuda
# TORCH_CUDA_ARCH_LIST="7.0 7.5 8.6+PTX" pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--build-option=--force_cuda" --config-settings="--build-option=--blas=openblas"

apt-get update && apt-get install -y libopenblas-dev&& \
     rm -rf /var/lib/apt/lists/*
export CUDA_HOME=/usr/local/cuda

cd 
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.6+PTX" python setup.py install --force_cuda --cuda_home=/usr/local/cuda --blas=openblas