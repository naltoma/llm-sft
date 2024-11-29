import xml.etree.ElementTree as ET
import json
import argparse

def convert_xml_to_json(input_file, output_file, translator="refA"):
    # XMLファイルをパース
    tree = ET.parse(input_file)
    root = tree.getroot()

    dataset = []

    # 各doc要素を処理
    for doc in root.findall(".//doc[@origlang='ja']"):
        # 原文を抽出
        src_segments = doc.find(".//src[@lang='ja']")
        if src_segments is None:
            continue
        
        # 翻訳文を抽出
        ref_segments = doc.find(f".//ref[@lang='en'][@translator='{translator}']")
        if ref_segments is None:
            continue

        # セグメントごとに処理
        for src_seg, ref_seg in zip(src_segments.findall(".//seg"), ref_segments.findall(".//seg")):
            src_text = src_seg.text.strip() if src_seg.text else ""
            ref_text = ref_seg.text.strip() if ref_seg.text else ""

            # データセット形式に変換
            dataset.append({
                "instruction": "Translate the following Japanese text into English.",
                "input": src_text,
                "output": ref_text
            })

    # JSONに保存
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=2)

    print(f"JSONデータセットが {output_file} に保存されました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XML to LLaMa-Factory JSON format")
    parser.add_argument("input_file", type=str, help="Input XML file path")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("--translator", type=str, default="refA", help="Translator to select (default: refA)")

    args = parser.parse_args()

    convert_xml_to_json(args.input_file, args.output_file, args.translator)

