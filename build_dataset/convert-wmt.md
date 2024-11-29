# オリジナルのデータセットを用意する例（翻訳コーパス）
(1) LLaMA-Factory/data フォルダ内に指定形式でデータセットを用意し、(2) llamafactory-cli が参照できるように dataset_info.json を編集する必要がある。以下では日英翻訳モデルを構築することを想定し、(1),(2)を例示している。

💡 **Tips**: 今回はシンプルに instruciton, input, output だけを設定しています。他にもimageを指定したり、conversation を指定することもできるようです。詳細は[LLaMA-Factory/data](https://github.com/hiyouga/LLaMA-Factory/tree/main/data)を参照ください。

---
## (1-1) データセット用意：学習データ
[WMT2022](https://www.statmt.org/wmt22/metrics/index.html)の DA (Direct Assesment) data をダウンロードし、"ja-en"（日英対訳コーパス）のみを利用する。

### step 1: 2020-da.csv から 2020-da-en-ja.csv を用意する
```shell
# 2020年の DA (direct assesment) data をダウンロードし、解凍。
% curl -O https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-da.csv.tar.gz
% tar xvf 2020-da.csv.tar.gz

# ファイル冒頭2行を確認
% head -2 2020-da.csv
lp,src,mt,ref,z_score,score,annotators
ps-en,راځئ اوس د هندويزم او اسلام ترمنځ گډ ټکي وپلټو او جاج يې واخلو,Let's search for the joint points between Hinduism and Islam and get the chicken.,Let's analyse the differences between Hinduism and Islam and discuss them.,-2.12568303633751,25.0,1

# CSVファイル補足説明
# - lp = language pair（言語ペア）
# - src = source（原文）
# - mt = 翻訳例。モデル出力（使わない）
# - ref = reference（参照文。教師データとして利用）
# - z_score, score, annotators = mtに対する評価やあのテータのラベル（使わない）

# ja-enだけを取得
% grep "ja-en" 2020-da.csv > 2020-da-en-ja.csv
```

### step 2: csvファイルから Alpaca format に変換
[convert-wmt2020da-llama.py](./convert-wmt2020da-llama.py)参照。instructionとして「Translate the Japanese text into English.」を指定し、src列をinput、ref列をoutputとして設定しています。

```shell
python convert-wmt2020da-llama.py 2020-da-en-ja.csv wmt2020da-en-ja.json
```

---
## (1-2) データセット用意：テストデータ
[WMT23 News Systems and Evaluations](https://github.com/wmt-conference/wmt23-news-systems/)から wmttest2023.ja-en.all.xml をダウンロードし、"ja-en"（日英対訳コーパス）のみを利用する。なお、このファイルは「同一文に対する複数の異なるLLMや人間による翻訳」が含まれている。今回は refA と付けられている翻訳文（人間による翻訳）を教師データとして利用することにする。


### step 1: xmlファイルをダウンロード
```shell
# wmttest2023.ja-en.all.xml をダウンロード。
curl -O https://github.com/wmt-conference/wmt23-news-systems/raw/refs/heads/master/xml/wmttest2023.ja-en.all.xml
```

### step 2: Alpaca format に変換
[convert-wmttest2023-llama.py](./convert-wmttest2023-llama.py)参照。

💡 **Tips**: [Alpaca formatの例はこちら](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#alpaca-format)。

```shell
python convert-wmttest2023-llama.py wmttest2023.ja-en.all.xml wmttest2023.ja-en.all.json --translator refA

# 引数説明
# - wmttest2023.ja-en.all.xml: ソースファイル
# - wmttest2023.ja-en.all.json: 出力ファイル名
# - --translator refA: 教師データの指定
```

---
## (2) data/dataset_info.json の編集
今回の例だと学習データとして `wmt2020da-en-ja.json`, テストデータとして `wmttest2023.ja-en.all.json` を **data フォルダに用意** した。これらを llamafactory-cli から参照できるように data/dataset_info.json に以下のように追加する。
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
