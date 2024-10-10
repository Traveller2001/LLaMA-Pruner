import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import concurrent.futures
from tqdm import tqdm  

_atte_weights = []
def attn_hook(module, input, output):
    attn_output, attn_weights, past_key_value = output
    global _atte_weights
    _atte_weights.append(attn_weights.detach().cpu())

def register_hooks(model):
    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(attn_hook)

def save_attention_head(attn_weights, output_dir, head, num_layers, cols, plot_last_row=False):
    rows = (num_layers + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for layer in range(num_layers):
        if plot_last_row:
            plot_data = attn_weights[layer][head, -1, :] 
            sns.lineplot(data=plot_data, ax=axes[layer])
        else:
            plot_data = attn_weights[layer][head] 
            sns.heatmap(plot_data, cbar=False, ax=axes[layer])
        axes[layer].set_title(f'Layer {layer+1}')

    for ax in axes[num_layers:]:
        ax.axis('off')
    plt.suptitle(f'Head {head+1}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'head_{head+1}.png'))
    plt.close()

def save_attention_heatmaps(attn_map, output_dir, plot_last_row=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_layers = len(attn_map)
    
    if len(attn_map[0].shape) == 4:
        num_heads = attn_map[0].shape[1]
        attn_weights = [attn_layer[0].detach().cpu().numpy() for attn_layer in attn_map]
    else:
        num_heads = attn_map[0].shape[0]
        attn_weights = [attn_layer.detach().cpu().numpy() for attn_layer in attn_map]
    
    cols = 4
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(save_attention_head, attn_weights, output_dir, head, num_layers, cols, plot_last_row) for head in range(num_heads)]
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=num_heads, desc="Processing attention heads"):
            pass  
        for future in concurrent.futures.as_completed(futures):
            future.result()  

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    register_hooks(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True)
    model.eval()
    prompt = "Hello AI World"
    prompt_path = os.path.join(args.output_dir, "prompt.json")
    with open(prompt_path, "w") as f:
        json.dump({"prompt": prompt}, f, ensure_ascii=False, indent=4)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model(input_ids=input_ids, output_attentions=True)
    attn_map = outputs.attentions
    if len(_atte_weights) > 0:
        save_attention_heatmaps(_atte_weights, args.output_dir, args.plot_last_row)
    else:
        save_attention_heatmaps(attn_map, args.output_dir, args.plot_last_row)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the attention heatmaps")
    parser.add_argument("--plot_last_row", action="store_true", help="Plot the last row of the attention map instead of the full heatmap")
    args = parser.parse_args()
    main(args)