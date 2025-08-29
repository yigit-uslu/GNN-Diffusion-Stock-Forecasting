#!/bin/bash
MAX_JOBS=1 # max number of parallel scripts allowed
True=1
False=0

# Set the script mode to train or test
SCRIPT="train"

CUDA_VISIBLE_DEVICES=0 # Set this to the GPU you want to use, e.g., 0, 1, 2, etc.

# group_name="17-Aug/100-nodes-graph-density-12-rmin-0.6"
group_name="28-Aug/UnitTests"



################################ GENERAL CONFIG ################################
################################################################################
random_seed=42

diffusion_policy="price-forecast"
# Set base project name to "Interference-Power" if diffusion_policy is "interference-power", otherwise set it to "Power-Allocation"
if [[ "$diffusion_policy" == "price-forecast" ]]; then  
    base_project_name="Price-Forecast" # "Power-Allocation"
elif [[ "$diffusion_policy" == "state-augmented-power-allocation" ]]; then
    base_project_name="Power-Allocation" # "Power-Allocation"
else
    base_project_name=""
fi

echo "Diffusion_policy is set to: $diffusion_policy"
echo "Base project name is set to: $base_project_name"

# base_project_name="Interference-Power" # "Power-Allocation"
base_root="./$base_project_name-GDM-Experiments/$group_name"


# export CUDA_VISIBLE_DEVICES=1
# sudo nvidia-cuda-mps-control -d  # Start MPS daemon
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE  # Each process gets 1/N th of GPU resources
# echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
export WANDB_API_KEY="b59bf4e20cad4116ea8d4641c24b7aabda909559"
export WANDB_RUN_GROUP=$group_name
# export WANDB_PROJECT="State-Augmented-Dual-Regression-Policies"
export WANDB_PROJECT="${base_project_name}-Diffusion-Learning"
export WANDB_WATCH="all"

track_run_with_wandb=True
log_dataset=True

################################################################################
################################ GENERAL CONFIG ################################





################################### ACCELERATOR CONFIG ###################################
##########################################################################################

gradient_accumulation_steps=1 # gradient accumulation steps
sync_each_batch=False # setting to True improves memory usage at the expense of speed
auto_device_placement=False

###########################################################################################
################################### ACCELERATOR CONFIG ####################################




################################ DATASET CONFIG ################################
################################################################################

dataset_name="S&P 100"
data_dir="./SP100AnalysisWithGNNs/data/SP100/raw"
corr_threshold=0.7 # Thresholding for the correlation matrix
sector_bonus=0.05 # Bonus for stocks sharing the same sector

################################################################################
################################ DATASET CONFIG ################################



#################### DIFFUSION MODEL & TRAINING CONFIG ####################
###########################################################################

gnn_backbone_diffusion="resplus-gnn"
###############################
lr_diffusion=1e-2 # 1e-3
weight_decay_diffusion=1e-4 # 1e-4
lr_sched_gamma_diffusion=0.8 # 0.5 # 0.9
pgrad_clipping_constant_diffusion=10.0
###############################
n_layers_diffusion=6 # 4
k_hops_diffusion=2
norm_layer_diffusion="graph" # "batch"
layer_norm_mode_diffusion="node"
conv_layer_normalize_diffusion=False # $sa_conv_layer_normalize
dropout_rate_diffusion=0.2
###############################
hidden_dim_diffusion=256 # 128
###############################
x_batch_size_diffusion=500 # 5000
diffuse_n_samples_diffusion=500
sampler_diffusion="ddpm"
batch_size_diffusion=1

###########################################################################
#################### DIFFUSION MODEL & TRAINING CONFIG ####################



################ PRE-EXPERIMENT NOTES ####################
load_cd_train_chkpt_path_diffusion="none" # "Power-Allocation-GDM-Experiments/09-Jul/100-nodes-graph-density-12-rmin-0.6/batch_size_diffusion-16/norm_layer_diffusion-layer/1752273515.731762_seed_2024/cd-models/cd_train_chkpt_epoch_399.pt" # "Power-Allocation-GDM-Experiments/07-Jul/100-nodes-graph-density-12-rmin-0.6/batch_size_diffusion-2/norm_layer_diffusion-layer/1752269408.9051833_seed_2024/cd-models/cd_train_chkpt_epoch_95.pt"


# num_graphs=80 # 32
MAX_EFFECTIVE_BATCH_SIZE_DIFFUSION=200
EFFECTIVE_TRAINING_STEPS=100000 #128000
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(expr 100 / $MAX_JOBS)

norm_layer_diffusion_values=("layer")
batch_size_diffusion_values=($batch_size_diffusion)


process_counter=0
for batch_size_diffusion in "${batch_size_diffusion_values[@]}"; do

    # x_batch_size_diffusion_temp=$(expr $MAX_EFFECTIVE_BATCH_SIZE_DIFFUSION / $batch_size_diffusion)
    # x_batch_size_diffusion_temp=$(expr $x_batch_size_diffusion_temp \* $gradient_accumulation_steps)
    # x_batch_size_diffusion_temp=$(expr $x_batch_size_diffusion_temp / 8)
    # x_batch_size_diffusion=$(expr 8 \* $x_batch_size_diffusion_temp)

    # diffuse_n_samples_diffusion_temp=$(expr $x_batch_size_diffusion / 20)
    # diffuse_n_samples_diffusion=$(expr $diffuse_n_samples_diffusion_temp \* 2)
    diffuse_n_samples_diffusion=100 # Set to 100 for now, as per the recent edits

    # n_iters_per_epoch=$(expr $num_graphs / $batch_size_diffusion)
    n_iters_per_epoch=10
    n_epochs_diffusion=$(expr $EFFECTIVE_TRAINING_STEPS / $n_iters_per_epoch)
    

    echo "[Diffusion] Batch_size: $batch_size_diffusion, x_batch_size: $x_batch_size_diffusion, diffuse_n_samples: $diffuse_n_samples_diffusion, n_epochs: $n_epochs_diffusion"
    
    for norm_layer_diffusion in "${norm_layer_diffusion_values[@]}"; do
        # root="$base_root/batch_size_diffusion-$batch_size_diffusion/norm_layer_diffusion-$norm_layer_diffusion" 
        root="$base_root"
        echo "Root: $root"

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --config_file ./accelerate_configs/config.yaml --num_processes 1 ./main_${SCRIPT}_diffusion_learner.py \
                                                                  --noaccelerate_diffusion=$False \
                                                                  --max_jobs=$MAX_JOBS \
                                                                  --track=$track_run_with_wandb --wandb_project_name="${WANDB_PROJECT}" --wandb_group_name=$group_name \
                                                                  --diffusion_policy="${diffusion_policy}" \
                                                                  --root="$root" --random_seed=$random_seed \
                                                                  --log_dataset=$log_dataset \
                                                                  --gradient_accumulation_steps=$gradient_accumulation_steps --sync_with_dataloader=$False --split_batches=$True --sync_each_batch=$sync_each_batch --auto_device_placement=$auto_device_placement \
                                                                  --dataset_name="${dataset_name}" --data_dir="${data_dir}" \
                                                                  --corr_threshold=$corr_threshold --sector_bonus=$sector_bonus \
                                                                  --load_cd_train_chkpt_path_diffusion="${load_cd_train_chkpt_path_diffusion}" \
                                                                  --gnn_backbone_diffusion="${gnn_backbone_diffusion}" \
                                                                  --batch_size_diffusion=$batch_size_diffusion \
                                                                  --load_model_chkpt_path_diffusion="none" \
                                                                  --x_batch_size_diffusion=$x_batch_size_diffusion --diffuse_n_samples_diffusion=$diffuse_n_samples_diffusion --sampler_diffusion="${sampler_diffusion}" \
                                                                  --lr_diffusion=$lr_diffusion --weight_decay_diffusion=$weight_decay_diffusion --lr_sched_gamma_diffusion=$lr_sched_gamma_diffusion \
                                                                  --hidden_dim_diffusion=$hidden_dim_diffusion --n_layers_diffusion=$n_layers_diffusion --k_hops_diffusion=$k_hops_diffusion \
                                                                  --n_epochs_diffusion=$n_epochs_diffusion \
                                                                  --pgrad_clipping_constant_diffusion=$pgrad_clipping_constant_diffusion \
                                                                  --norm_layer_diffusion="${norm_layer_diffusion}" --layer_norm_mode_diffusion="${layer_norm_mode_diffusion}" --conv_layer_normalize_diffusion=$conv_layer_normalize_diffusion \
                                                                  --dropout_rate_diffusion=$dropout_rate_diffusion  --pool_ratio_diffusion=0.8 --apply_gcn_norm_diffusion=$True &

                                                            
    ((process_counter++))  # Increment job count

    # If max jobs are running, wait for them to finish before launching more
    if (( process_counter >= MAX_JOBS )); then
        wait  # Wait for all background jobs to complete
        process_counter=0  # Reset counter after waiting 
    fi
   
    done

done

wait