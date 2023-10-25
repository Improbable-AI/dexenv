#!/bin/bash

set -euxo pipefail
pip install --ignore-installed PyYAML
pip install scipy ipython cython matplotlib gym \
    opencv-python tensorboard gitpython pillow cloudpickle \
    colorlog gputil imageio ninja h5py orjson seaborn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install trimesh iopath fvcore hydra-core open3d
pip install recommonmark rl-games wandb python-dotenv
pip install ffmpeg-python
pip install numpy==1.23.5
pip uninstall setuptools -y
pip install setuptools==59.5.0