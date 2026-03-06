#!/bin/bash

# initialize.sh
# This is an initialization script for the entire project.
# Sets allthe fields present in the .installed.json present in PackageDownloadModule
# and installs the basic dependencies to run the project in a project venv. 

echo -e "[FILE INTIALIZATION...]\n"

tr -d '\n' < ./ConfigurationModule/ConfigFiles/supported_devices_library.json | \
sed 's/.*\[\(.*\)\].*/\1/' | \
tr ',' '\n' | \
sed 's/[" ]//g' | \
awk 'BEGIN {print "{"} {printf "%s  \"%s\": false", (NR>1?",\n":""), $1} END {print "\n}"}' \
> ./PackageDownloadModule/requirementsFileDirectory/.installed.json

echo -e "File .installed.json set!\n"

mkdir ModelData/Dataset
mkdir ModelData/ONNXModels 
mkdir Results
echo -e "[DATASET EXTRACTION...]\n"


unzip -q casting_products.zip; rm -rf casting_512x512; rm -rf ./ModelData/Dataset/casting_data; mv -f casting_data/casting_data ./ModelData/Dataset/; rm -rf casting_data


echo -e "DATASET CORRECTLY EXTRACTED!\n" 


echo -e "[BASE DEPENDENCIES INSTALLATION...]\n"

rm -rf ./Converters/CoralConverter/venv || true; 
rm -rf ./Converters/FusionConverter/venv || true; 
rm -rf venv || true; python3.10 -m venv venv
source venv/bin/activate

python3.10 -m pip install -e . --config-setting editable_mode=compat


echo "PROJECT CORRECTLY INITIALIZED."
echo "To use the CLI, activate the environment: source venv/bin/activate"
echo "Then run: margis or margis -h"
