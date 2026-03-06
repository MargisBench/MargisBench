#!/bin/bash

# This script allows to start a Docker Container to compile the Models for the Fusion target. 
# The scripts requires one single argument, the CONFIG_ID of the configuration. This is useful to 
# gather the correct .onnx models. The container is executed with some folder bindings (HefModels/CONFIG_ID, ONNXModels/CONFIG_ID) 
# and file bindings. Moreover, it is executed with the needed pipeline of command to compile the files. 


IMAGE_NAME="margisbench/margisbench_hailo_compiler:latest"
CONTAINER_NAME="hailo8_compiler2"

if [[ -n $1 ]]; then
    CONFIG_ID=$1
    if [[ -d "./ModelData/ONNXModels/${CONFIG_ID}" ]]; then
        echo "./ModelData/ONNXModels/${CONFIG_ID} EXISTS..."
        echo ""
    else
       echo "Directory not found. Exiting."
       exit 1
    fi
else
    echo "Incorrect usage."
    echo "Usage: ./docker_hailo_compiler.sh <config-id-number-dir>"
    exit 1
fi

DOCKER_ARGS="--privileged
             -v ./ModelData/ONNXModels/${CONFIG_ID}:/app/ONNXModels \
             -v ./Converters/FusionConverter/HefModels/${CONFIG_ID}:/app/HefModels:rw \
             -v ./Converters/FusionConverter/HefModels/compile.sh:/app/HefModels/compile.sh \
             -v ./Converters/FusionConverter/HefModels/create_memory_profiling.sh:/app/HefModels/create_memory_profiling.sh \
             -v ./Converters/FusionConverter/HefModels/model_modifications.py:/app/HefModels/model_modifications.py \
             -v ./Converters/FusionConverter/Calibration/CalibrationArrays:/app/CalibrationArrays:rw 
            "

docker run --rm $DOCKER_ARGS --name $CONTAINER_NAME $IMAGE_NAME /bin/sh -c 'cd HefModels/ && ./compile.sh && ./create_memory_profiling.sh'
#docker run --rm -ti $DOCKER_ARGS --name $CONTAINER_NAME $IMAGE_NAME 
