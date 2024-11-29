# LLaMA-Factory チュートリアル

## LLaMA-Factory とは何か？
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)は `Easy and Efficient LLM Fine-Tuning` とあるように、LLMをファインチューニングする（以下SFT）ためのフレームワークです。多くのLLMに対し様々な学習手法を統一的に実行することができます。対応しているモデルや代表的な学習手法については[Readmeのfeatures](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#features)を参照してください。

---
## 想定環境
ここでは[学科システム](https://ie.u-ryukyu.ac.jp/syskan/server_configuration/)のように、以下の環境でLLMをSFTすることを想定しています。

- ハードウェア構成: intel CPU + Nvidia GPU
  - [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)でGPU見えてる状態になっていること。
- ソフトウェア環境
  - Apptainer（旧Singularity）: see also, [Singularityのすゝめ](https://ie.u-ryukyu.ac.jp/syskan/service/singularity/)
  - Slurm: see also, [Slurmについて](https://ie.u-ryukyu.ac.jp/syskan/service/slurm/)

💡 **Tips**: 個人のデスクトップPCで実行するならば、venv, pyenv, poetry等で仮想環境構築して実行すると良いでしょう。

---
## コンテナ作成
Apptainerでdefinitionファイル（[llama-factory.def](./llama-factory.def)）を使って構築します。
```shell
# 実行例
apptainer build llama-facotry.sif llama-factory.def
# `llama-facotry.sif` は作成するsifファイル名。自由に付けて良い。
# `llama-facotry.def` は設定ファイル名。
```

definitionファイルでやっていることは以下の通り。

- クリーンなUbuntu24.04を用意。
- [Python環境構築ガイド 〉 Ubuntu環境のPython](https://www.python.jp/install/ubuntu/index.html)を参考に開発環境を用意し、git, Python 3, python3-pipの最新版をインストール。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)の環境構築。
- [wandb](https://wandb.ai/site/ja/)もインストール。

他にも追加しておきたいパッケージがあれば definition ファイルを修正するか、一度作成した sif ファイルをベースに追加インストールして構築し直すと良いでしょう。

---
## 例一覧
- [例1: LLaMA-Factory の実行例を試してみる](./ex1-train-tutorial.md)
- [例2: オリジナルデータでSFT (LoRA) する](./ex2-train-originaldata.md)
- [例3: テストデータでモデルを評価する](./ex3-eval-testdata.md)
- [例4: LLMをベンチマークで総合評価する](./ex4-eval-mmlu.md)
