import pandas as pd
import json
import argparse

def convert_to_json(input_csv, output_json):
    # Load the CSV file
    columns = ['lp', 'src', 'mt', 'ref', 'z_score', 'score', 'annotators']
    data = pd.read_csv(input_csv, names=columns, header=None)
    
    # Convert to JSON format
    json_data = []
    for _, row in data.iterrows():
        entry = {
            "instruction": "Translate the Japanese text into English.",
            "input": row["src"],
            "output": row["ref"]
        }
        json_data.append(entry)
    
    # Save to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Conversion complete. JSON saved to {output_json}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Convert CSV to JSON for LLaMA Factory.")
    parser.add_argument('input_csv', type=str, help="Path to the input CSV file")
    parser.add_argument('output_json', type=str, help="Path to the output JSON file")
    args = parser.parse_args()
    
    # Call the conversion function
    convert_to_json(args.input_csv, args.output_json)
