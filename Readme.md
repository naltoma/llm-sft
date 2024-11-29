# LLM SFT Tips
## 基本方針
### SFT方針
- 細かなカスタマイズなしにSFTするだけならば、コマンドライン実行するだけで済ませたい。

### 環境構築方針
いくつかの代表的な環境を想定し、環境毎に実行方法を整理する予定。2024年11月29日時点ではケース1のみ整理。

- ケース1: Intel CPU + Nvidia GPU 環境
    - [学科サーバ](https://ie.u-ryukyu.ac.jp/syskan/server_configuration/)想定。
        - [Apptainer（旧Singularity）](https://github.com/apptainer/apptainer)でコンテナ作成し、[Slurm](https://ie.u-ryukyu.ac.jp/syskan/opening-introduction/singularity-slurm.html#1)でジョブ実行。
    - 全体の流れとチュートリアルはこちら => [./llama-factory/Readme.md](./llama-factory/Readme.md)
- ケース2: Apple Sillicon環境（GPUなし）
    - 手元の macbook air でお試し実行することを想定。[MLX LM](https://github.com/ml-explore/mlx-examples)かな？
    - llama factoryでオプション調整するだけでも良いかもしれない。
        - [Will LLaMa-Factory run/train on a MacBook Pro M3 Max with 128GB ? #2329](https://github.com/hiyouga/LLaMA-Factory/issues/2329)
    - ＊まだ試してない＊
