#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

IMAGE=improbableailab/dexenv:latest

while getopts i: flag
do
    case "${flag}" in
        i) image=${OPTARG};;
    esac
done

if ! [ -z "${image}" ]
then 
    IMAGE=${image}
fi

wandb docker-run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$PWD/../../:/workspace/" \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --net=host \
    ${IMAGE}









