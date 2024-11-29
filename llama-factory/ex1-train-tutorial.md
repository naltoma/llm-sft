## 例1: LLaMA-Factory の実行例を試してみる
### step 1. apptainer exec 時の設定
[公式 readme の quickstart](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#quickstart)にあるように、基本的には llamafactory-cli に yaml 形式で用意した設定ファイルを指定して実行します。ここで注意する点は以下の通りです。

- 予め LLaMA-Factory リポジトリのクローンを用意しておく。
- `llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml` のように実行する際には、カレントディレクトリを LLaMA-Factory リポジトリのトップに移動しておく。

これらを apptainer exec 時に設定する必要があります。例えば以下のようになります。

```shell
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
```

### yamlファイルの補足説明
llama3_lora_sft.yamlでは以下のような設定をしているようです。いろんな例が[公式サイトexamplesフォルダ](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md)にあるので、参考にすると良いでしょう。（全パラメータの説明一覧はないかも？）

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct # モデルの指定

### method
stage: sft # 学習ステージの指定。pt(pre-train), dpo等が指定できる模様。
do_train: true # 学習するためのフラグ。
finetuning_type: lora # 学習方法。full, freeze, lora,,
lora_target: all # lora対象

### dataset
dataset: identity,alpaca_en_demo # データセット名。dataset_info.jsonに登録する必要あり。
template: llama3 # モデル毎にテンプレートが用意されているっぽい。
cutoff_len: 2048 # 打ち切りトークン数
max_samples: 1000 # 最大サンプル数。全てを使いたい場合には -1 を指定。
overwrite_cache: true
preprocessing_num_workers: 16 # データ前処理に使用するワーカー数 = スレッド数

### output
output_dir: saves/llama3-8b/lora/sft # 学習結果等の保存先
logging_steps: 10 # 10ステップ毎にログ出力
save_steps: 500 # モデルを保存するステップ頻度（500ステップ毎）
plot_loss: true # 訓練中にloss出力
overwrite_output_dir: true # output_dirを上書きする

### train
per_device_train_batch_size: 1 # 1ステップあたりに処理するバッチサイズ（サンプル数）
gradient_accumulation_steps: 8 # 勾配の累積ステップ数。
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine # 学習率スケジューラー。
warmup_ratio: 0.1 # ウォームアップステップの割合。全体の10%をウォームアップに使う。
bf16: true # 16ビット浮動小数点（bf16）形式を使用
ddp_timeout: 180000000 # DDP（分散データ並列処理）のタイムアウト時間

### eval
val_size: 0.1 # 検証データセットの割合
per_device_eval_batch_size: 1 # 検証時のバッチサイズ
eval_strategy: steps # 検証のタイミング
eval_steps: 500 # 検証を行う頻度（500ステップ毎）
```

### step 2. ジョブ投入用ファイルの作成
- 参考: [llamafactory-lora-sft.sbatch](./llamafactory-lora-sft.sbatch)
- 補足説明
  - 基本的には apptainer コマンドを実行しているだけです。それ以外の部分は Slurm 用の設定です。ここでは logs 以下にログ出力するように指定しているため、**事前に logs ディレクトリを作成しておく** 必要があります。
  - job-name, output, error は必要に応じて修正してください。

### step 3. Slurmにジョブ投入
```shell
# 事前に logs フォルダを作成
mkdir logs

# ジョブ投入
sbatch llamafactory-lora-sft.sbatch

# ジョブ処理状況確認
squeue

# ログファイル観察
tail -f logs/llama3_lora_sft.yaml-JOBID.err
tail -f logs/llama3_lora_sft.yaml-JOBID.out
```

### 実行結果（精製されるファイル一覧）
正常終了すると、yamlで指定した出力先（今回は LLaMA-Factory/save/llama3-8b/llora/sft-wmt）に実行結果が保存される。

- `README.md`: サマリ。base_model, 学習オプション, データセット, 最終loss, ライブラリのバージョン等が記入されている。
- `adapter_config.json`: アダプター層（LoRA, QLoRA等）の設定ファイル。
- `adapter_model.safetensors`: アダプター層の学習済み重みを保存したファイル。
- `all_results.json`: epoch, eval_loss (検証データに対する損失), train_loss, それらに要した時間が記録されている。
- `checkpoint-xxx/`: 特定のステップ（xxxステップ目）のモデルのチェックポイント。途中段階のモデルを利用したい場合に使う。（最終モデルだけが必要ならば不要）
- `eval_results.json`: all_results.jsonの一部。
- `special_tokens_map.json`: 特殊トークン（[CLS]、[SEP]、[PAD]など）情報。
- `tokenizer.json`: トークナイザの全ての情報（ボキャブラリやマージルール）。
- `tokenizer_config.json`: トークナイザの設定ファイル（例: ボキャブラリサイズ、トークン化のパラメータ）
- `train_results.json`: all_results.jsonの一部。
- `trainer_log.jsonl`: 学習の詳細なログがJSON Lines形式で記録。
  - 今回の例だと次のようなログが10ステップ毎に記録されている。
    - `{"current_steps": 10, "total_steps": 375, "loss": 2.203, "lr": 2.6315789473684212e-05, "epoch": 0.0796812749003984, "percentage": 2.67, "elapsed_time": "0:09:01", "remaining_time": "5:29:08"}`
- `trainer_state.json`: Trainer（学習管理ツール）の現在の状態（例: ステップ数、学習率スケジュール、その他の設定）。学習を再開する際に利用。
- `training_args.bin`: トレーニングの設定（例: 学習率、バッチサイズ、エポック数）がバイナリ形式で保存。何故バイナリ形式で保存されているのかは不明。
- `training_loss.png`: 学習中の損失値の変化を可視化したプロット画像。

💡 **Tips**
- `checkpoint-xxx/` は学習途中のモデルが保存されています。途中のモデルが不要ならば削除しておくと良いでしょう。（ストレージ容量食うので）
