## 例4: LLMをベンチマークで総合評価する
loss, blue, rougeのような指標ではふんわりとした性能しか測れません。LLaMA-Factoryでは、指標毎のデータセットを複数用意し、多方面にわたる性能評価（Measuring Massive Multitask Language Understanding; MMLU）もコマンドライン一発で実行することができます。この例では `meta-llama/Meta-Llama-3-8B-Instruct` と、これを例2でSFTしたモデルを mmlu で評価してみます。

💡 **Tips**:
- [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval) も参考になるでしょう。
- MMLU: [Measuring Massive Multitask Language Understanding, 2021](https://arxiv.org/abs/2009.03300)

### step 1: yaml設定ファイルの作成
LLaMA-Factory/examples/train_lora/llama3_lora_eval.yaml を複製して、オリジナルデータを利用するように以下の点を修正。修正版は[llama3_eval_mmlu.yaml](./llama3_eval_mmlu.yaml)を参照。編集内容は以下の通り。

- アダプター指定を削除。（meta-llama/Meta-Llama-3-8B-Instructそのものを評価対象とする）
- loraしていないので削除。
- 出力先を変更。
- バッチサイズを変更。
    - 4だとメモリオーバーで途中で止まってしまう。

```shell
% diff llama3_lora_eval.yaml llama3_eval_mmlu.yaml
3,6d2
< adapter_name_or_path: saves/llama3-8b/lora/sft
< 
< ### method
< finetuning_type: lora
15c11
< save_dir: saves/llama3-8b/lora/eval
---
> save_dir: saves/llama3-8b/lora/eval-llama3-8B-Instruct
18c14
< batch_size: 4
---
> batch_size: 1
```

### step 2: ジョブ投入用ファイルの作成
- 参考: [llamafactory-eval-mmlu.sbatch](./llamafactory-eval-mmlu.sbatch)
- 例1のように yaml ファイルを指定している。

### step 3: Slurmにジョブ投入
```shell
sbatch llamafactory-eval-mmlu.sbatch
```

### 実行結果精製されるファイル一覧（評価結果）
output_dirで指定したディレクトリにに以下のファイルが出力される。

- `results.json`: 詳細ログ。ベンチマーク毎に問題idと出力結果が記録されている模様。
- `results.og`: MMLU以下のスコアが記録されている。詳細は[MMLU論文](https://arxiv.org/abs/2009.03300)を読もう。
    - Average: 総合評価。
    - STEM: 科学、技術、工学、数学分野
    - Social Sciences: 社会科学分野
    - Humanities: 人文学分野
    - Other: その他

以下は例2で作成したモデルとの比較結果です。どのカテゴリでも少しずつ劣化していますね。一般的にファインチューニングは特定タスクに特化したモデルを作ることが多いため、その対象タスクにおいては性能向上しやすいです。しかしそれ以外のタスクにおいては劣化してしまうことがあります。

| Category        | Meta-Llama-3-8B-Instruct | 例2でSFTしたモデル |
|:----------------|----------------:|----------------:|
| Average         |           65.92 |           64.93 |
| STEM            |           56.49 |           56.03 |
| Social Sciences |           76.47 |           76.02 |
| Humanities      |           60.94 |           59.17 |
| Other           |           71.93 |           71.04 |
