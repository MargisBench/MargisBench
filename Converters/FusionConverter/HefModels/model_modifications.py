"""
This script is used in a Docker Container in order to perform the Optimization (Quantization)
of the .har files of the model in the conversion pipeline. 

It takes the input parameters, search for target layers that have to be reduced in dimension
(due to Hailo Accelerator Architecture that allows a represenation shift of 2 for some operations), 
and performs the optimization step with a specific ModelScript. 

Parameters
----------

- Model Name
- Har File Path
- Calib Set Path
- Quantized Har Path
"""

from hailo_sdk_client import ClientRunner, CalibrationDataType
import sys
import numpy as np
import re


if (len(sys.argv) != 5):
    print("Incorrect Usage.")
    print("Usage: python3.10 model_modifications.py <model-name> <har_file_path> <calib-set-path> <quantized-har-path>")
    exit(1)

model_name=sys.argv[1]
har_path=sys.argv[2]
calib_data_path=sys.argv[3]
quantized_har_path=sys.argv[4]


runner = ClientRunner(har=har_path)

all_layer_names=[layer.name for layer in runner.get_hn_model().stable_toposort()]


target_layers = ["avgpool1", "avgpool2", "avgpool_op"]

model_script="normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"
model_script+="performance_param(compiler_optimization_level=0)\n"


existing_layers = []
for layer_name in all_layer_names:
    parts= layer_name.split("/")
    if any(p in target_layers for p in parts):
        existing_layers.append(layer_name)


if len(existing_layers) !=0:
    print("[INFO] AT LEAST A TARGET LAYER IS PRESENT!")
    print(f"[INFO] EXISTING LAYERS: {existing_layers}")
    model_script+=f"pre_quantization_optimization(global_avgpool_reduction, layers=[{', '.join(existing_layers)}], division_factors=[2, 2])\n"



runner.load_model_script(model_script)
runner.optimize(calib_data_path, data_type=CalibrationDataType.npy_file)
runner.save_har(quantized_har_path)
print(f"[INFO] MODEL {har_path} SUCESSFULLY QUANTIZED!")

