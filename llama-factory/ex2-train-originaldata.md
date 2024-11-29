## 例2: オリジナルデータでSFT (LoRA) する
ここでは日英翻訳タスクを例に、オリジナルデータを用意してSFTする流れを説明します。なおここでは全パラメータ更新によるSFTではなく、低ランク行列を挿入して学習する LoRA で SFT しています。

💡 **Tips**:
- LoRAについては例えば[NVIDIA TensorRT-LLM による、LoRA LLM のチューニングとデプロイ](https://developer.nvidia.com/ja-jp/blog/tune-and-deploy-lora-llms-with-nvidia-tensorrt-llm/)が参考になるでしょう。
- LoRAで学習した場合、モデルは「オリジナルモデル＋アダプター層」という形で構築され、オリジナルモデルのパラメータは凍結した状態で利用します。このため学習結果として保存するのはアダプター層のみで良くなり、ストレージにも優しいです。
    - 例えばここでやっているアダプター層の重みは adapter_model.safetensors に保存されており、これはたかだか 81MB です。

### step 1: データセットの用意
llamafactory-cli が参照できるようにするためには、(1) LLaMA-Factory/data 以下に Alpaca Format でデータを用意し、(2) data/dataset_info.json を編集する必要があります。

#### Alpaca Formatでデータを用意
ここでは学習データセットを[WMT2020のDAデータセット 2020-da.csv.tar.gz](https://www.statmt.org/wmt22/metrics/index.html)から "ja-en" のみを利用することにします。またテストデータセットとして[WMT23 News Systems and Evaluations](https://github.com/wmt-conference/wmt23-news-systems/)から "ja-en" のみを利用することにします。これらのデータの準備方法は [../build_dataset/convert-wmt.md](../build_dataset/convert-wmttest2023-llama.md)を参照ください。

#### data/dataset_info.jsonを編集
基本的には以下のようにインデックスを付けてファイル名を指定するぐらいで良さそうです。必要に応じて他データセットの例（dataフォルダ参照）を参考にカスタマイズしていくことになるでしょう。

```JSON
{
  "wmt2020da-en-ja": {
    "file_name": "wmt2020da-en-ja.json"
  },
  "wmttest2023.ja-en.all": {
    "file_name": "wmttest2023.ja-en.all.json"
  },
  //省略
}
```

### step 2: yaml設定ファイルの作成
LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml を複製して、オリジナルデータを利用するように以下の点を修正。修正版は[llama3_lora_sft_wmt.yaml](./llama3_lora_sft_wmt.yaml)を参照。編集内容は以下の通り。

- データセットを変更。
- trainデータの max_samplesを変更。
  - [loader.py](https://github.com/hiyouga/LLaMA-Factory/blob/00031b1a66ade1c2665ce7a069a756cccbcb07f1/src/llamafactory/data/loader.py#L146)を眺めると `max_samples = min(data_args.max_samples, len(dataset))` としているので、実数より大きな値を設定しておけば全サンプルを使うことになる模様。
- 出力先を変更。
- バッチサイズを変更。
  - 💡 **Tips**: バッチサイズは2^Nで調整することが多い。ハードウェアやモデルのデータ型によって最適値は異なる模様。例えばfp16データ型なら64の倍数がベターらしい。
    - 参考: [Batch size choice](https://huggingface.co/docs/transformers/ja/perf_train_gpu_one#batch-size-choice)
- wandbを設定。（別途WANDB_API_KEYを設定する必要あり）

```shell
% diff llama3_lora_sft.yaml llama3_lora_sft_wmt.yaml
11c11
< dataset: identity,alpaca_en_demo
---
> dataset: wmt2020da-en-ja
14c14
< max_samples: 1000
---
> max_samples: 10000
19c19
< output_dir: saves/llama3-8b/lora/sft
---
> output_dir: saves/llama3-8b/lora/sft-wmt
26c26
< per_device_train_batch_size: 1
---
> per_device_train_batch_size: 8
39a40,43
> 
> ### additional
> report_to: wandb
> run_name: llama3_lora_sft_wmt # sample
```

💡 **Tips**: 今回は設定していませんが、LoRAでは用意する低ランク行列のランク数を64にすると特定タスクへの性能と元々の汎化性能とのバランスが良いという報告（[LoRA vs Full Fine-tuning: An Illusion of Equivalence, 2024](https://arxiv.org/abs/2410.21228)）があるようです。より正確には、ランク1ではそれ以前に学んだことを忘却しやすい傾向があるようです（同論文 Figure 8）。ランク8で若干抑えられるようですが、より大きなランク程忘却しにくくなるようです。ランク数を設定するには `lora_rank: 64` のように記述すると良さそう（多分）。

### step 3: ジョブ投入用ファイルの作成
- 参考: [llamafactory-lora-sft-wmt.sbatch](./llamafactory-lora-sft-wmt.sbatch)
- 例1との違いは llama3_lora_sft_wmt.yaml を指定していることだけ。

### step 4: Slurmにジョブ投入
```shell
sbatch llamafactory-lora-sft-wmt.sbatch
```

### 結果の参照
output_dirで指定したディレクトリに保存されています。詳細は[例1の「実行結果精製されるファイル一覧」](./ex1-train-tutorial.md#実行結果精製されるファイル一覧)を参照。
