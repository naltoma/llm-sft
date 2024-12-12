## 例3: テストデータでモデルを評価する
ここでは例2で構築した日英翻訳タスクモデルを対象に、オリジナルデータを用意して評価する流れを説明します。

### step 1: データセットの用意
[../build_dataset/convert-wmt.md](../build_dataset/convert-wmttest2023-llama.md)を参照。

### step 2: yaml設定ファイルの作成
LLaMA-Factory/examples/train_lora/llama3_lora_predict.yaml を複製して、オリジナルデータを利用するように以下の点を修正。修正版は[llama3_lora_wmt-predict.yaml](./llama3_lora_wmt-predict.yaml)を参照。編集内容は以下の通り。

- 構築したモデルが保存されている場所を変更。
- テストデータセットを変更。
- テストに用いるデータ数を全数に指定。（trainでも同じ指定できそうなんだけど、あっちはNGだった）
- 出力先を変更。

```shell
% diff llama3_lora_predict.yaml llama3_lora_wmt-predict.yaml
3c3
< adapter_name_or_path: saves/llama3-8b/lora/sft
---
> adapter_name_or_path: saves/llama3-8b/lora/sft-wmt
11c11
< eval_dataset: identity,alpaca_en_demo
---
> eval_dataset: wmttest2023.ja-en.all
14c14
< max_samples: 50
---
> max_samples: 10000
19c19
< output_dir: saves/llama3-8b/lora/predict
---
> output_dir: saves/llama3-8b/lora/predict-wmt
```

SFTとの違いという点では、以下の点が大切なようです。

- do_predict: true
- eval_dataset: wmttest2023.ja-en.all
- per_device_eval_batch_size: 1
- predict_with_generate: true
- 💡 **Tips**
  - `max_samples`については、-1とすると全サンプルになった気がするのだけど、今はダメな模様。実サンプル数より大きなサイズを指定しておけば、全サンプルを利用してくれます。
    - 参考: [data/loader.py:_load_single_dataset()](https://github.com/hiyouga/LLaMA-Factory/blob/581392fdd1d7aca39558e817350a90e7392162a8/src/llamafactory/data/loader.py#L148)

### step 3: ジョブ投入用ファイルの作成
- 参考: [llamafactory-lora-predict-wmt.sbatch](./llamafactory-lora-predict-wmt.sbatch)
- 例1, 例2との違いは llama3_lora_wmt-predict.yaml を指定していることだけ。

### step 4: Slurmにジョブ投入
```shell
sbatch llamafactory-lora-predict-wmt.sbatch
```

### 実行結果精製されるファイル一覧（評価結果）
output_dirで指定したディレクトリにに以下のファイルが出力される。

- `all_results.json`: BLEU-4, ROUGE-1, ROUGE-2, ROUGE-l等、教師データに対するスコア。
- `generated_predictions.jsonl`: テスト時の詳細ログがJSON Lines形式で記録。
  - 今回の例だと次のようなログが指定したサンプル毎に記録されている。
    - `{"prompt": "user\n\nTranslate the following Japanese text into English.\n徹底した感染対策／クラツー公式 - 助成金でお得旅assistant\n\n", "label": "Thorough Infection Controls / Club Tourism Official - Subsidized Budget Travel", "predict": "Thorough Infection Prevention Measures / Krats Tour Official - Enjoy a discounted trip with a subsidy"}`
- `predict_results.json`: all_results.jsonと同じ。
- `trainer_log.jsonl`: 指定ステップ毎のログ。
  - `{"current_steps": 5, "total_steps": 50, "percentage": 10.0, "elapsed_time": "0:00:04", "remaining_time": "0:00:36"}`
