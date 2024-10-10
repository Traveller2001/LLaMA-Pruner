export CUDA_VISIBLE_DEVICES=0
model_name=Llama-3-8B
python src/hook_attn_map.py \
 -m /141nfs/zhangqingyu_17/hf_models/llama-2-7b \
 -o /141nfs/zhangqingyu_17/code/LLaMA-Pruner/result/attn_map/$model_name/last_row \