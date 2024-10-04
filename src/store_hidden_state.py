from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from  argparse import ArgumentParser
from loguru import logger
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

dtype_dict = {
    'fp32':torch.float32,
    'fp16':torch.float16,
    'bf16':torch.bfloat16
}
CONTEXT=1024
STRIDE=256
_hooked_hidden_states = []

def input_state_hook(module, input, output):
    global _hooked_hidden_states
    _hooked_hidden_states.append((input[0].detach()))



def output_hidden_state_hook(module, input, output):
     _hooked_hidden_states.append((output[0].detach()))

def register_model_hook(model):
    for idx, layer in enumerate(model.model.layers):
        if idx == 0:
            layer.register_forward_hook(input_state_hook)
        layer.register_forward_hook(output_hidden_state_hook)

    
if __name__ == "__main__":
    parser = ArgumentParser("hook_hidden_states from model.")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dtype', required=False, default='fp32', choices=list(dtype_dict.keys()))
    parser.add_argument('--device', required=False, default='cuda:0')
    parser.add_argument('--calib_set', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    logger.info(f'args: {args}')

    logger.info(f'loading tokenizer from {args.model_path} ...')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True, 
        add_bos_token=False, 
        add_eos_token=False,
        use_fast = False)
    logger.info("tokenizer loaded")

    logger.info(f'loading model from {args.model_path} ...')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 trust_remote_code=True,  
                                                 torch_dtype=dtype_dict[args.dtype],
                                                 device_map="cuda:0",
                                                #  use_flash_attention_2="flash_attention_2",
                                            )
    logger.info("model loaded")

    register_model_hook(model)
    
    test_splits = open(args.calib_set).readlines()
    logger.add('log/store_hidden_state.log')
    layerwise_hidden_states = None
    sample_cnt = 0
    for sample_id, sample in tqdm(enumerate(test_splits), total=len(test_splits)):
        sample = json.loads(sample)
        input_ids = tokenizer(sample['text'], truncation=False, return_tensors="pt").input_ids
        input_ids = input_ids
        for i in tqdm(range(0, input_ids.shape[1]-CONTEXT, STRIDE),total=(input_ids.shape[1]-CONTEXT) //STRIDE):
            if i >= STRIDE*10:
                break
            tmp_input = input_ids[:,i:i+CONTEXT].to(model.device)
            with torch.no_grad():
                pred = model.forward(input_ids=tmp_input)
                sample_cnt += 1
                hooked_hidden_states = torch.vstack(_hooked_hidden_states)
                _hooked_hidden_states = []
                if layerwise_hidden_states is None:
                    layerwise_hidden_states = hooked_hidden_states
                else:
                    layerwise_hidden_states += hooked_hidden_states

    logger.info(f'sample number:{sample_cnt}')
    layerwise_hidden_states /= sample_cnt
    save_path = os.path.join(args.output_dir,f"{args.model_name}/hidden_state_pg19.pt")
    logger.info(f'saveing hidden_states to {save_path}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(layerwise_hidden_states,save_path)
    
    