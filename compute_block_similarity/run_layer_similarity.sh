#!/bin/bash

# This Bash script runs the Python script with arguments

# Run the Python script with command-line arguments
python layer_similarity.py --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
                      --dataset "arcee-ai/sec-data-mini" \
                      --dataset_column "text" \
                      --batch_size 8 \
                      --max_length 1024 \
                      --layers_to_skip 8 \
                      --dataset_size 4000 \
                      --dataset_subset "train" 