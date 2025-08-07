#!/bin/bash
set -e

cd /workspace/IsaacGymEnvs
pip install -e .

cd /workspace/isaacgym/python
pip install -e .

cd /workspace/dexenv
pip install -e .

echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc

echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

## Install ROS Foxy
apt update && apt install locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

apt install software-properties-common
add-apt-repository universe

apt update && apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

apt update

apt install --no-install-recommends -y ros-foxy-desktop python3-argcomplete

apt install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev

apt install -y ros-dev-tools
apt install -y ros-foxy-sensor-msgs-py
apt install ros-foxy-rmw-cyclonedds-cpp

pip install "pin<2.9.0"
pip install meshcat

eval "bash"

exec "$@"