#!/bin/bash

apt-get update && apt-get install -y vulkan-utils mesa-vulkan-drivers mesa-common-dev && \
     rm -rf /var/lib/apt/lists/*

rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json
