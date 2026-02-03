#!/bin/bash

# =========================================================================
# Hyperparameter Tuning Script for PGA-KD (Training + Eval + WandB Sync)
# =========================================================================

# --- Configuration ---
NUM_GPUS_PER_NODE=8
TRAIN_SCRIPT="main.py"
EVAL_SCRIPT="eval_mmeb.py"
BASE_EXP_NAME="PGA_Tune_Full_CLS"
WANDB_PROJECT="vlm_distillation" 

# 1. Dataset Configuration
USE_FULLSET=true
LORA_R=64
BATCH_SIZE=32

if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")
    echo "Running with FULL dataset set."
else
    SUBSETS=("ImageNet_1K")
    echo "Running with SINGLE dataset (ImageNet_1K) for tuning efficiency."
fi

# Evaluation Subsets (Datasets to appear in WandB Table)
EVAL_SUBSETS_ARR=("ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")

# =========================================================================
# Hyperparameter Grid
# =========================================================================

PGA_MSE_LOSS_WEIGHTS=(0.3)
PGA_SCL_LOSS_WEIGHTS=(0.5 0.1)
PGA_LOSS_WEIGHTS=(0.5 1.0)
PGA_SPECTRAL_VARIANCE_THRESHOLDS=(0.85 0.8)

# =========================================================================
# Tuning Loop
# =========================================================================

for mse_w in "${PGA_MSE_LOSS_WEIGHTS[@]}"; do
  for scl_w in "${PGA_SCL_LOSS_WEIGHTS[@]}"; do
    for pga_w in "${PGA_LOSS_WEIGHTS[@]}"; do
      for var_t in "${PGA_SPECTRAL_VARIANCE_THRESHOLDS[@]}"; do

        # 1. Generate Experiment Name
        # Naming convention updated to reflect new PGA params
        CURRENT_EXP_NAME="${BASE_EXP_NAME}_pga${pga_w}_scl${scl_w}_mse${mse_w}_var${var_t}"
        OUTPUT_DIR="training/$CURRENT_EXP_NAME"
        
        # 2. Generate and Export Unique WandB ID
        RAND_ID=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
        export WANDB_RUN_ID="PGA_${pga_w}_${scl_w}_${mse_w}_$(date +%Y%m%d_%H%M%S)_${RAND_ID}"
        export WANDB_PROJECT="$WANDB_PROJECT"

        echo "================================================================"
        echo "Starting Experiment: $CURRENT_EXP_NAME"
        echo "Run ID: $WANDB_RUN_ID"
        echo "Params -> PGA: $pga_w | SCL: $scl_w | MSE: $mse_w | Var: $var_t"
        echo "================================================================"

        # 3. Run Training
        # Updated arguments to match PGAKDLoss config
        torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
            --model_name "apple/FastVLM-0.5B" \
            --teacher_model_name "raghavlite/B3_Qwen2_2B" \
            --lora True \
            --teacher_lora True \
            --lora_r $LORA_R \
            --teacher_lora_r 8 \
            --teacher_pooling "eos" \
            --teacher_backbone "qwen2_vl" \
            --model_backbone "llava_qwen2" \
            --pooling "eos" \
            --dataset_name "TIGER-Lab/MMEB-train" \
            --eval_dataset_name "TIGER-Lab/MMEB-eval" \
            --subset_name "${SUBSETS[@]}" \
            --eval_subset_name "${EVAL_SUBSETS_ARR[@]}" \
            --dataset_split "original" \
            --image_dir "vlm2vec_train/MMEB-train" \
            --eval_image_dir "eval_images/" \
            --output_dir "$OUTPUT_DIR" \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --num_train_epochs 1 \
            --bf16 \
            --save_total_limit 1 \
            --logging_steps 1 \
            --save_strategy "epoch" \
            --seed 42 \
            --weight_decay 0.01 \
            --normalize True \
            --teacher_normalize True \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.03 \
            --projector_config_path "./config/projector_config.json" \
            --projector_lr 5e-4 \
            --image_resolution "low" \
            --report_to "wandb" \
            --run_name "$CURRENT_EXP_NAME" \
            --kd_loss_type "pga" \
            --pga_mse_loss_weight "$mse_w" \
            --pga_loss_weight "$pga_w" \
            --pga_scl_loss_weight "$scl_w" \
            --pga_spectral_variance_threshold "$var_t"

        # 4. Run Evaluation (On checkpoint-final)
        # Find the latest checkpoint directory (handling cases where explicit 'checkpoint-final' might not exist)
        CHECKPOINT_PATH=$(ls -d $OUTPUT_DIR/checkpoint-* | sort -V | tail -n 1)
        
        if [ -z "$CHECKPOINT_PATH" ]; then
             echo "Warning: No checkpoint found in $OUTPUT_DIR. Defaulting to output dir."
             CHECKPOINT_PATH="$OUTPUT_DIR"
        fi
        
        echo "--> Training finished. Starting Evaluation on: $CHECKPOINT_PATH"
        
        accelerate launch --multi_gpu --num_processes="$NUM_GPUS_PER_NODE" $EVAL_SCRIPT \
            --model_name "$CHECKPOINT_PATH" \
            --encode_output_path "$OUTPUT_DIR" \
            --dataset_name "TIGER-Lab/MMEB-eval" \
            --subset_name "${EVAL_SUBSETS_ARR[@]}" \
            --dataset_split "test" \
            --per_device_eval_batch_size 128 \
            --image_dir "eval_images/" \
            --pooling "eos" \
            --model_backbone "llava_qwen2" \
            --normalize True \
            --bf16 \
            --tgt_prefix_mod 

        # 5. Log Eval Results to WandB
        echo "--> Evaluation finished. Pushing results to WandB..."
        python3 scripts/log_eval_results.py "$WANDB_RUN_ID" "$WANDB_PROJECT" "$OUTPUT_DIR"

        echo "Finished Cycle: $CURRENT_EXP_NAME"
        
        # 6. Cleanup (Optional)
        # rm -rf "$OUTPUT_DIR/checkpoint-*"

      done
    done
  done
done

echo "All tuning experiments completed."