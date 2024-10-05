
#!/bin/bash
MODEL_NAME="llama-2-7b"
MODEL_PATH="~/hf_models/$MODEL_NAME"
RESULT_PATH="~/LLaMA-Pruner/result"

# python pruner.py \
#     --model_path $MODEL_PATH \
#     --metadata_path "$MODEL_PATH/model.safetensors.index.json" \
#     --safetensor \
#     --output_dir "$OUTPUT_DIR" \


LAYERS_TO_REMOVE="27 26 25 28 24 29 23 21 22"


TOTAL_LAYERS=32
REMAINING_LAYERS=$((TOTAL_LAYERS - $(echo $LAYERS_TO_REMOVE | wc -w)))


OUTPUT_DIR="$RESULT_PATH/$MODEL_NAME-${REMAINING_LAYERS}layer"
mkdir -p "$OUTPUT_DIR"

python src/pruner.py \
    --model_path "$MODEL_PATH" \
    --metadata_path "$MODEL_PATH/pytorch_model.bin.index.json" \
    --output_dir "$OUTPUT_DIR" \
    --layers_to_remove $LAYERS_TO_REMOVE \
    --total_layers $TOTAL_LAYERS
