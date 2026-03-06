#!/bin/bash

# This script allows to perform the convertion from ONNX Models to HEF files in three steps: Parsing, Optimization (Quantizing) and Compile. 
# The Parsing is performed through hailo compiler CLI tool.
# The Optimization is performed through the model_modifications.py script
# The Compiling is performed through the hailo compiler CLI tool.

 

echo "[INFO - PARSING]"
echo ""
for file in ../ONNXModels/*.onnx; do
    

    filename=$(echo $file | cut -d "." -f 3 | cut -d "/" -f 3)

    if [ -f ./har_files/${filename}.har ]; then
        echo "[INFO] FILE ${file} ALREADY PARSED!"
        continue
    fi

    echo ""
    echo "[INFO] PARSING ${file} ..."

    hailo parser onnx --hw-arch hailo8 --har-path ./har_files/${filename}.har $file
    rm -f *.log
done

echo ""
echo "[INFO - OPTIMIZATION]"
echo ""
for file in ./har_files/*.har; do
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)

    if [ -f har_files_quantized/${filename}_quantized.har ]; then
        echo "[INFO] FILE ${file} ALREADY OPTIMIZED!"
        continue
    fi

    echo ""
    echo "[INFO] QUANTIZING ${file} ..."
    #This regex help to extract the name of the model to construct the .npy paths.
    #The first part prints the filename and the lines from the folder CalibrationArrays
    #The second command performs a longest prefix match between the first string and each line took by the ls command, removes "pruned" or "distilled" and removes the hypotethically final _, prints the lines
    
    calib_set_filename="$(printf "%s\n%s\n" $filename "$(ls ../CalibrationArrays/)" | sed -n '1h; 1!{G; s/^\(.*\).*\n\1.*$/\1/; s/_$//; p}' | sort | tail -n 1)_calibration_data.npy"
    echo ""
    echo "CALIBRATION SET FILENAME: ${calib_set_filename}"
    #hailo optimize --hw-arch hailo8 --output-har-path "./har_files_quantized/${filename}_quantized.har" --calib-set-path "../CalibrationArrays/${calib_set_filename}" --model-script "./avgpool1.alls" $file 
    python3.10 model_modifications.py $filename $file ../CalibrationArrays/${calib_set_filename} har_files_quantized/${filename}_quantized.har
    rm -f *.log
done

echo ""
echo "[INFO - COMPILE]"
echo ""
for file in ./har_files_quantized/*.har; do
    
    filename=$(echo $file | cut -d "." -f 2 | cut -d "/" -f 3)
    model_name=$(printf "%s\n" "$filename" | sed 's/_quantized//g')

    if [ -f ./hef_files/${model_name}.hef ]; then
        echo "[INFO] FILE  ${file} ALREADY COMPILED!"
        continue
    fi

    echo ""
    echo "[INFO] COMPILING ${file} ..."

    hailo compiler --hw-arch hailo8 --output-dir ./hef_files $file 
    rm -f *.log ./hef_files/*.log 
done

