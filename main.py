from overspecifier import Overspecifier
import json
import pandas as pd

import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_row(row_data, args_dict):
    idx, row = row_data
    prompt = row['Example']

    overspec = Overspecifier(
        prompt=prompt,
        model=args_dict['model'],
        max_attrs_per_object=args_dict['max_attrs_per_object'],
        max_values_per_attribute=args_dict['max_values_per_attribute'],
        contrasting=True,
        seed=args_dict['seed']
    )

    tokens = overspec.generate_token_structure(verbose=False)
    result = {
        "prompt": prompt,
        "attr_per_token": [token.model_dump() for token in tokens]
    }
    return (idx, result, prompt)

def main():
    parser = argparse.ArgumentParser(description="Tokenize and analyze prompts from a CSV file with overspecifier.")
    parser.add_argument("--csv_path", type=str, default="csv/tc1.csv", help="Path to the input CSV file containing prompts")
    parser.add_argument("--output_path", type=str, default="results/tc1.jsonl", help="Path to output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="OpenAI model to use")
    parser.add_argument("--max_attrs_per_object", type=int, default=3, help="Max attributes per object")
    parser.add_argument("--max_values_per_attribute", type=int, default=5, help="Max values per attribute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=3, help="Limit the number of rows to process from CSV")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (processes)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    rows = df.head(args.limit)

    args_dict = {
        'model': args.model,
        'max_attrs_per_object': args.max_attrs_per_object,
        'max_values_per_attribute': args.max_values_per_attribute,
        'seed': args.seed,
    }

    rows_iterable = list(rows.iterrows())
    results_buffer = [None] * len(rows_iterable)

    with open(args.output_path, 'w') as out, ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i, row_data in enumerate(rows_iterable, 1):
            futures.append(executor.submit(process_row, row_data, args_dict))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Expanding Attributes..."):
            try:
                row_idx, result, prompt = future.result()
                results_buffer[row_idx] = json.dumps(result) + '\n'
            except Exception as e:
                tqdm.write(f"Error in processing row: {e}")

        for jsonl_line in results_buffer:
            if jsonl_line:
                out.write(jsonl_line)
                out.flush()

if __name__ == "__main__":
    main()

