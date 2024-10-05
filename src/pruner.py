import json
import argparse
from safetensors.torch import load_file, save_file
import glob
import os
import torch
import shutil

def load_model(args, file_path):
    if args.safetensor:
        return load_file(file_path)
    else:
        return torch.load(file_path, map_location='cpu', weights_only=True)


def load_model(args, file_path):
    if args.safetensor:
        return load_file(file_path)
    else:
        return torch.load(file_path, map_location='cpu', weights_only=True)

def copy_files(src_dir, dst_dir, patterns, exclude_patterns=None):
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(src_dir, pattern)):
            if exclude_patterns and any(file_path.endswith(ep) for ep in exclude_patterns):
                continue
            shutil.copy2(file_path, dst_dir)
            print(f"Copied {os.path.basename(file_path)} to {dst_dir}")

def update_config_json(config_path, remaining_layers):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'num_hidden_layers' in config:
        config['num_hidden_layers'] = remaining_layers
        print(f"Updated num_hidden_layers to {remaining_layers}")
    else:
        print("num_hidden_layers not found in config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model .safetensors files')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the metadata JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the new model and metadata files')
    parser.add_argument('--total_layers', type=int, required=True, help='Total number of layers')
    parser.add_argument('--layers_to_remove', type=int, nargs='+', required=True, help='List of layer indices to remove')
    parser.add_argument('--safetensor', action='store_true')
    args = parser.parse_args()

    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

    model = {}
    for weight_key, weight_file in metadata['weight_map'].items():
        model.update(load_model(args, os.path.join(args.model_path, weight_file)))
    
    num_layers = args.total_layers  # total number of layers
    remaining_layers = num_layers - len(args.layers_to_remove)

     # Filter out the layers to remove
    filtered_model = {k: v for k, v in model.items() if not any(k.startswith(f'model.layers.{i}.') for i in args.layers_to_remove)}

    # Remap the remaining layers
    new_model = {}
    new_weight_map = {}
    layer_idx = 0
    for i in range(num_layers):
        if i not in args.layers_to_remove:
            for key, value in filtered_model.items():
                if key.startswith(f'model.layers.{i}.'):
                    new_key = key.replace(f'model.layers.{i}.', f'model.layers.{layer_idx}.')
                    new_model[new_key] = value
                    new_weight_map[new_key] = f"model-{remaining_layers}layers-00001-of-00001.safetensors"
            layer_idx += 1
    
    # Add other non-layer weights to the new model
    for key, value in model.items():
        if not key.startswith('model.layers.'):
            new_model[key] = value
            new_weight_map[key] = f"model-{remaining_layers}layers-00001-of-00001.safetensors"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the new model
    output_model_path = os.path.join(args.output_dir, f"model-{remaining_layers}layers-00001-of-00001.safetensors")
    meta_data = {"format": "pt"}
    save_file(new_model, output_model_path, metadata=meta_data)

    # Update the metadata
    new_metadata = {
        "metadata": metadata["metadata"],
        "weight_map": new_weight_map
    }
    output_metadata_path = os.path.join(args.output_dir, f"model-{remaining_layers}layers.safetensors.index.json")
    with open(output_metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=2)

    print(f"Model and metadata saved to {output_model_path} and {output_metadata_path}")
    
    # Copy JSON files (excluding *index.json) and tokenizer.model from model_path to output_dir
    copy_files(args.model_path, args.output_dir, ['*.json', 'tokenizer.model'], ['index.json'])

    # Update config.json
    config_path = os.path.join(args.output_dir, "config.json")
    if os.path.exists(config_path):
        update_config_json(config_path, remaining_layers)
    else:
        print("config.json not found in the output directory")


if __name__ == "__main__":
    main()