import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel 
console = Console()

def print_rich_output(bi_list):
    generated_text = "\n".join([f"Layer {layer_id}: BI = {bi}" for layer_id, bi in bi_list])
    panel = Panel(generated_text, title="Sorted BI Values", expand=False)
    console.print(panel)

model_name = "llama-2-7b"
length = 1024
domain = "pg19"
saved_file_path = f'~/{model_name}/hidden_state_{domain}.pt'

loaded_data = torch.load(saved_file_path)


hs = loaded_data
hs = [torch.tensor(h) for h in hs]


all_layer_bi = []

for i in range(len(hs)-1):
    hs_in = hs[i].squeeze(0)
    hs_out = hs[i+1].squeeze(0)
    
    cosine_sim = F.cosine_similarity(hs_in, hs_out, dim=1).mean().item()
    all_layer_bi.append((i, 1-cosine_sim))
    

sorted_bi = sorted(all_layer_bi, key=lambda x: x[1])
print_rich_output(sorted_bi)
