#!/bin/bash

set -euxo pipefail
pip install iopath fvcore

pip install "git+https://github.com/facebookresearch/pytorch3d.git"