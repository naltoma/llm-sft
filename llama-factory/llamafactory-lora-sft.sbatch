#!/bin/bash
#SBATCH --job-name llama3_lora_sft.yaml
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

apptainer exec \
	--bind /fullpath/to/your/LLaMA-Factory:/mnt \
	--nv \
    ~/path/to/your/sif-image \
	bash -c "cd /mnt \
    && llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml"

# llamafactory-cli オプション説明
# (a) --bind /fullpath/to/your/LLaMA-Factory:/mnt
#    /fullpath/to/your/LLaMA-Factory を /mnt としてマウントする。
# (b) --nv # GPU利用
# (c) ~/path/to/your/sif-image
#    apptainerで実行するコンテナへの相対パス（フルパスでも良い）
# (d) bash -c "cd /mnt
#     bash -c "A && B && C" で、Aを実行→Bを実行→Cを実行となる。コマンドを複数列挙できる。
#     /mntへ移動
# (e) && llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
#     学習設定ファイルllama3_lora_sft.yamlを指定して学習を実行。
#     データセットの場所を相対パスで書いているため、参照できるディレクトリに移動してから実行する必要がある。
