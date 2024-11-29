#!/bin/bash
#SBATCH --job-name llama3_lora_wmt-predict.yaml
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

apptainer exec \
	--bind /fullpath/to/your/LLaMA-Factory:/mnt \
	--nv \
    ~/path/to/your/sif-image \
	bash -c "cd /mnt \
	&& llamafactory-cli train examples/train_lora/llama3_lora_wmt-predict.yaml"
