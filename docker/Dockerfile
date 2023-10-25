FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# setup timezone
RUN echo 'America/New_York' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Replacing shell with bash for later docker build commands
RUN mv /bin/sh /bin/sh-old && \
    ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    xvfb \
    tmux \
    libgtk2.0-dev \
    ca-certificates \
    libjpeg-dev \
    freeglut3-dev  \
    wget \
    vim \
    software-properties-common \
    libopenblas-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


COPY ./install_python_3_8.sh /tmp/install_python_3_8.sh
RUN chmod +x /tmp/install_python_3_8.sh && /tmp/install_python_3_8.sh
COPY ./install_py_packages.sh /tmp/install_py_packages.sh
RUN chmod +x /tmp/install_py_packages.sh && /tmp/install_py_packages.sh
COPY ./install_pytorch3d.sh /tmp/install_pytorch3d.sh
RUN chmod +x /tmp/install_pytorch3d.sh && /tmp/install_pytorch3d.sh
COPY ./install_mk.sh /tmp/install_mk.sh
RUN chmod +x /tmp/install_mk.sh && /tmp/install_mk.sh

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

COPY ./gym_rendering.sh /tmp/gym_rendering.sh
RUN chmod +x /tmp/gym_rendering.sh && /tmp/gym_rendering.sh
COPY ./nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY ./10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
RUN rm -rf /tmp/* && mkdir /install_dir
WORKDIR /install_dir
COPY ./entrypoint.sh /install_dir/entrypoint.sh
WORKDIR /workspace
ENTRYPOINT ["bash", "/install_dir/entrypoint.sh"]

CMD ["bash"]