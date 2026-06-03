#!/usr/bin/env bash
set -euo pipefail

# devices & launcher
export CUDA_VISIBLE_DEVICES="0,1,2,3"
LAUNCH_CMD="accelerate launch --config_file configs/deepspeed_config.yaml --debug train.py"

# fixed args
COMMON_ARGS=(
  --seed 100
  --dataset_name "Kamyar-zeinalipour/recall-assistant-dataset"
  --chat_template_format "none"
  --add_special_tokens False
  --append_concat_token False
  --splits "train,test"
  --max_seq_len 1600
  --num_train_epochs 3
  --logging_steps 5
  --log_level "info"
  --logging_strategy "steps"
  --eval_strategy "epoch"
  --push_to_hub False
  #--hub_private_repo True
  #--hub_strategy "every_save"
  --bf16 True
  --packing True
  --learning_rate 1e-4
  --lr_scheduler_type "cosine"
  --weight_decay 1e-4
  --warmup_ratio 0.0
  --max_grad_norm 1.0
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 2
  --gradient_checkpointing True
  --use_reentrant False
  --dataset_text_field "content"
  --use_flash_attn True
  --use_peft_lora True
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.1
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,embed_tokens,lm_head"
  --use_4bit_qunatization False
)

# list of models
MODELS=(





#"mistralai/Mistral-Small-24B-Instruct-2501"
#"Qwen/Qwen2.5-14B-Instruct"
"mistralai/Mistral-Nemo-Instruct-2407"
 "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"

  
)

# fractions to sweep
#FRACTIONS=(0.25 0.5 0.75 1)
FRACTIONS=(1)

for MODEL in "${MODELS[@]}"; do
  # sanitize model name for filesystem
  SANITIZED_MODEL=${MODEL//\//_}
  for FRAC in "${FRACTIONS[@]}"; do
    OUTDIR="outputs/${SANITIZED_MODEL}_frac${FRAC}_hint_32_64"
    mkdir -p "$OUTDIR"

    echo "==> Training $MODEL with ${FRAC} of data → $OUTDIR"
    $LAUNCH_CMD \
      --model_name_or_path "$MODEL" \
      --train_fraction "$FRAC" \
      --output_dir "$OUTDIR" \
      "${COMMON_ARGS[@]}"
  done
done
