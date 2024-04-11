import logging
import csv
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets

from utils import get_last_non_padded_tokens, compute_block_distances
from typing import Optional

logging.basicConfig(level=logging.INFO)

# Set seed
torch.manual_seed(42)
np.random.seed(42)


def main(model_path: str, dataset: str, dataset_column: str, batch_size: int, max_length: int,
         layers_to_skip: int, dataset_size: Optional[int] = None, dataset_subset: Optional[str] = "eval"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if resource is a problem
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(model_path,  
                                                 device_map="auto", 
                                                 quantization_config=quantization_config, 
                                                 output_hidden_states=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataset = datasets.load_dataset(dataset, split=dataset_subset)
    if dataset_size:
        dataset = dataset.select(range(dataset_size))

    dataloader = DataLoader(dataset[dataset_column], batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize a list to store distances for each block across the dataset
    all_distances = [[] for _ in range(model.config.num_hidden_layers - layers_to_skip)]


    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = tokenizer(batch, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states
        last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention_mask)

        # Remove the first element to account for the input layer not being considered a model hidden layer
        # This adjustment is necessary for analyses focusing on the model's internal transformations
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
        
        # Ensure that the length of last_non_padded_hidden_states matches the number of model hidden layers minus one
        assert len(last_non_padded_hidden_states) == model.config.num_hidden_layers, "Length of last_non_padded_hidden_states  \
        does not match expected number of hidden layers."

        # Compute distances and append to all_distances
        distances = compute_block_distances(last_non_padded_hidden_states, layers_to_skip)
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

    # Calculate average distances for each block
    average_distances = [np.mean(block_distances) for block_distances in all_distances]

    # Write the average distances to a CSV file and compute the minimum average distance
    min_distance = float('inf')  # Initialize with infinity
    min_distance_layer = 0  # Initialize with an impossible value

    with open('layer_distances.csv', 'w', newline='') as csvfile:
        fieldnames = ['block_start', 'block_end', 'average_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, avg_dist in enumerate(average_distances):
            # Write each row to the CSV
            writer.writerow({
                'block_start': i + 1,  # layer indices are 1-based in the paper
                'block_end': i + 1 + layers_to_skip,
                'average_distance': avg_dist
            })
            
            if avg_dist < min_distance:
                min_distance = avg_dist
                min_distance_layer = i + 1  

    # Log the layer with the minimum average distance
    logging.info(f"Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} has the minimum average distance of {min_distance}. Consider examining this layer more closely for potential optimization or removal.")
    logging.info("Layer distances written to layer_distances.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model analysis.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--dataset_column", type=str, required=True, help="The specific column of the dataset to use.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of the tokenized input.")
    parser.add_argument("--layers_to_skip", type=int, required=True, help="Number of layers to skip.")
    parser.add_argument("--dataset_size", type=int, help="Optional argument to specify the size of the dataset.")
    parser.add_argument("--dataset_subset", type=str, default="eval", help="Subset of the dataset to use (e.g., 'train', 'eval').")
    parser.add_argument("--device", type=str, help="Device to run the model on ('cpu', 'cuda').")

    args = parser.parse_args()

    main(args.model_path, args.dataset, args.dataset_column, args.batch_size,
         args.max_length, args.layers_to_skip, args.dataset_size, args.dataset_subset)
