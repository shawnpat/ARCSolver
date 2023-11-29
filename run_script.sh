#!/bin/bash

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q scipy

# Initialize a counter
counter=0
# Define the directory containing the input data files
dir="/workspace/data/ARC/evaluation/"
# Loop through the input files
for file in "$dir"*
do
    counter=$((counter+1))
    for i in {1..10}
    do
        python clear_gpu.py
    done
    echo "Running inference on file number $counter: $file"
    python test_v001.py "$file"
done
