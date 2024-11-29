#!/bin/bash
#SBATCH --job-name llama3_eval_mmlu.yaml
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

apptainer exec \
	--bind /home/teacher/tnal/temp/llamafactory/LLaMA-Factory:/mnt \
	--nv ~/SIF-images/llama-factory20241122.sif \
	bash -c "cd /mnt \
	&& llamafactory-cli eval examples/train_lora/llama3_eval_mmlu.yaml
	"
