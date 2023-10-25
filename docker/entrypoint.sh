#!/bin/bash
set -e

cd /workspace/IsaacGymEnvs
pip install -e .

cd /workspace/isaacgym/python
pip install -e .


cd /workspace/dexenv
pip install -e .

eval "bash"

exec "$@"