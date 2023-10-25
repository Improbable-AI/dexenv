# Visual Dexterity

---

This is the codebase for [Visual Dexterity: In-Hand Reorientation of Novel and Complex Object Shapes](https://arxiv.org/abs/2211.11744), accepted by Science Robotics. While we provide the code that uses the D'Claw robot hand, it can be easily adapted to other robot hands.

### [[Project Page]](https://taochenshh.github.io/projects/visual-dexterity), [[arXiv]](https://arxiv.org/abs/2211.11744)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10039110.svg)](https://doi.org/10.5281/zenodo.10039110)


## :books: Citation

```
@article{chen2022visual,
    title={Visual Dexterity: In-Hand Reorientation of Novel and Complex Object Shapes},
    author={Chen, Tao and Tippur, Megha and Wu, Siyang and Kumar, Vikash and Adelson, Edward and Agrawal, Pulkit},
    journal={arXiv preprint arXiv:2211.11744},
    year={2022}
}
```

```
@article{chen2021system,
    title={A System for General In-Hand Object Re-Orientation},
    author={Chen, Tao and Xu, Jie and Agrawal, Pulkit},
    journal={Conference on Robot Learning},
    year={2021}
}
```

## :gear: Installation

#### Dependencies
* [PyTorch](https://pytorch.org/)
* [PyTorch3D](https://pytorch3d.org/)
* [Isaac Gym](https://developer.nvidia.com/isaac-gym) (results in the paper are trained with Preview 3.)
* [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)
* [Wandb](https://wandb.ai/site)


#### Download packages
You can either use a virtual python environment or a docker for training. Below we show the process to set up the docker image. If you prefer using a virtual python environment, you can just install the dependencies in the virtual environment.

Here is how the directory looks like:
```
-- Root
---- dexenv
---- IsaacGymEnvs
---- isaacgym
```

```
# download packages
git clone git@github.com:taochenshh/dexenv.git
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git

# download IsaacGym from: 
# (https://developer.nvidia.com/isaac-gym)
# unzip it in the current directory

# remove the package dependencies in the setup.py in isaacgym/python and IsaacGymEnvs/
```

#### Download the assets

Download the robot and object assets from [here](https://drive.google.com/file/d/187F8caz8unBnrSpU5Bp1PrnKnNUWJ7SG/view?usp=sharing), and unzip it to `dexenv/dexenv/`.

#### Build the docker image
```
cd dexenv/docker
python docker_build.py -f Dockerfile
```

#### Launch the docker image

To run the docker image, you would need to have the nvidia-docker installed. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
# launch docker
./run_image.sh # you would need to have wandb installed in the python environment
```

In another terminal
```bash
./visualize_access.sh
# after this, you can close it, just need to run this once after every machine reboot
```


## :scroll: Usage

#### :bulb: Training Teacher

```bash
# if you are running in the docker, you might need to run the following line
git config --global --add safe.directory /workspace/dexenv

# debug teacher (run debug first to make sure everything runs)
cd /workspace/dexenv/dexenv/train/teacher
python mlp.py -cn=debug_dclaw # show the GUI
python mlp.py task.headless=True -cn=debug_dclaw # in headless mode

# if you wanna just train the hand to reorient a cube, add `task.env.name=DClawBase`
python mlp.py task.env.name=DClawBase -cn=debug_dclaw 

# training teacher
cd /workspace/dexenv/dexenv/train/teacher
python mlp.py -cn=dclaw
python mlp.py task.task.randomize=False -cn=dclaw # turn off domain randomization
python mlp.py task.env.name=DClawBase task.task.randomize=False -cn=dclaw # reorient a cube without domain randomization

# if you wanna change the number of objects or the number of environments
python mlp.py alg.num_envs=4000 task.obj.num_objs=10 -cn=dclaw

# testing teacher
cd /workspace/dexenv/dexenv/train/teacher
python mlp.py alg.num_envs=20 resume_id=<wandb exp ID> -cn=test_dclaw
# e.g. python mlp.py alg.num_envs=20 resume_id=dexenv/1d1tvd0b -cn=test_dclaw

```

#### :high_brightness: Training Student with Synthetic Point Cloud (student stage 1)

```
# debug student
cd /workspace/dexenv/dexenv/train/student
python rnn.py -cn=debug_dclaw_fptd

# training student
cd /workspace/dexenv/dexenv/train/student
python rnn.py -cn=dclaw_fptd

# testing student
cd /workspace/dexenv/dexenv/train/student
python rnn.py resume_id=<wandb exp ID> -cn=test_dclaw_fptd
```

#### :tada: Training Student with rendered Point Cloud (student stage 2)

```
# debug student
cd /workspace/dexenv/dexenv/train/student
python rnn.py -cn=debug_dclaw_rptd

# training student
cd /workspace/dexenv/dexenv/train/student
python rnn.py -cn=dclaw_rptd

# testing student
cd /workspace/dexenv/dexenv/train/student
python rnn.py resume_id=<wandb exp ID> -cn=test_dclaw_rptd
```

## :rocket: Pre-trained models

We provide the pre-trained models for both the teacher and the student (stage 2) in `dexenv/expert/artifacts`. The models were trained using Isaac Gym preview 3.

```
# to see the teacher pretrained model
cd /workspace/dexenv/dexenv/train/teacher
python demo.py

# to see the student pretrained model
cd /workspace/dexenv/dexenv/train/student
python rnn.py alg.num_envs=20 task.obj.num_objs=10  alg.pretrain_model=/workspace/dexenv/dexenv/expert/artifacts/student/train-model.pt test_pretrain=True test_num=3 -cn=debug_dclaw_rptd
```

