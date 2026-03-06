#!/bin/bash

# This script is used to clean the caches for the Generic Platform. 
# This is called through the function cleanCaches() defined in Utils.utilsFunctions,
# that's called between DoE's runs to gather indipendent measurements. 
# 
# First of all, it calles sync to save the dirty pages to memory, 
# then it puts 3 in /proc/sys/vm/drop_caches to drop caches, d-entries
# and inodes. 

/usr/bin/sync
echo 3 |sudo tee /proc/sys/vm/drop_caches
