#!/bin/bash

# =========================================================================
# Hyperparameter Tuning Script for SSA (Spectral-Structural Alignment)
# =========================================================================

# GPU Configuration
NUM_GPUS_PER_NODE=8
TRAIN_SCRIPT="main.py"
BASE_EXP_NAME="SSA_Tune_DocVQA"

# 1. Dataset Configuration
# Set to 'true' for full training, 'false' for faster tuning on DocVQA only
USE_FULLSET=false

if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W")
    echo "Running with FULL dataset set."
else
    SUBSETS=("DocVQA")
    echo "Running with SINGLE dataset (DocVQA) for tuning efficiency."
fi

# Evaluation Subsets (Datasets to appear in WandB)
EVAL_SUBSETS=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W") 

# =========================================================================
# Hyperparameter Grid
# =========================================================================

# A. KD Weight: Balance between Contrastive Loss and SSA Loss
# Range: [0.5, 2.0]
#   0.5: Prefer Student's contrastive learning
#   2.0: Force Student to mimic Teacher structure strictly
KD_WEIGHTS=(0.3 0.5 1.0)

# B. Spectral Variance Threshold: Amount of Teacher's energy (variance) to keep
#   0.90: Aggressive filtering (removes 10% noise/tail) - Good if Teacher is very different size
#   0.95: Balanced (Standard Espresso/PCA approach)
#   0.99: Conservative (Keeps almost all Teacher details)
VAR_THRESHOLDS=(0.90 0.95 0.99)

# C. Modality Gap Weight: Strength of the geometric distance alignment
#   0.1: Weak constraint
#   1.0: Strong constraint (Standard)
GAP_WEIGHTS=(0.1 1.0)

# D. Matryoshka Dimensions: Slicing for efficiency
#   ""  : Full dimension only (Standard SSA)
#   "64": Optimize for both Full and first 64 dimensions
MATRYOSHKA_OPTS=("64" "64 128") 

# =========================================================================
# Tuning Loop
# =========================================================================

for kd_w in "${KD_WEIGHTS[@]}"; do
  for var_t in "${VAR_THRESHOLDS[@]}"; do
    for gap_w in "${GAP_WEIGHTS[@]}"; do
      for mat_dim in "${MATRYOSHKA_OPTS[@]}"; do

        # 1. Generate Unique Experiment Name
        # Format: SSA_Tune_DocVQA_kd1.0_var0.95_gap1.0_mat64
        MAT_SUFFIX=""
        if [ -n "$mat_dim" ]; then
            MAT_SUFFIX="_mat${mat_dim}"
        fi
        
        CURRENT_EXP_NAME="${BASE_EXP_NAME}_kd${kd_w}_var${var_t}_gap${gap_w}${MAT_SUFFIX}"
        
        echo "================================================================"
        echo "Starting Experiment: $CURRENT_EXP_NAME"
        echo "  > KD Weight: $kd_w"
        echo "  > Variance Threshold: $var_t"
        echo "  > Gap Weight: $gap_w"
        echo "  > Matryoshka Dims: ${mat_dim:-None}"
        echo "================================================================"

        # 2. Prepare Optional Arguments
        MAT_ARG=""
        if [ -n "$mat_dim" ]; then
            MAT_ARG="--ssa_matryoshka_dims $mat_dim"
        fi

        # 3. Run Training
        torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
            --model_name "apple/FastVLM-0.5B" \
            --teacher_model_name "raghavlite/B3_Qwen2_2B" \
            --lora True \
            --teacher_lora True \
            --lora_r 2 \
            --teacher_lora_r 8 \
            --teacher_pooling "eos" \
            --teacher_backbone "qwen2_vl" \
            --model_backbone "llava_qwen2" \
            --pooling "eos" \
            --dataset_name "TIGER-Lab/MMEB-train" \
            --eval_dataset_name "TIGER-Lab/MMEB-eval" \
            --subset_name "${SUBSETS[@]}" \
            --eval_subset_name "${EVAL_SUBSETS[@]}" \
            --dataset_split "original" \
            --image_dir "vlm2vec_train/MMEB-train" \
            --eval_image_dir "eval_images/" \
            --output_dir "training/$CURRENT_EXP_NAME" \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-5 \
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
            --projector_lr 5e-5 \
            --image_resolution "low" \
            --report_to "wandb" \
            --run_name "$CURRENT_EXP_NAME" \
            \
            --kd_loss_type "ssa" \
            --kd_weight "$kd_w" \
            --spectral_variance_threshold "$var_t" \
            --modality_gap_weight "$gap_w" \
            $MAT_ARG

        echo "Finished: $CURRENT_EXP_NAME"
        
        # Optional: Disk Cleanup (Remove checkpoints, keep only logs/wandb)
        # echo "Cleaning up checkpoints..."
        # rm -rf "training/$CURRENT_EXP_NAME/checkpoint-*"

      done
    done
  done
done