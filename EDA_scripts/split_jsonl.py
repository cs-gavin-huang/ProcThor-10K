import json
import os

input_file = 'train/train.jsonl'
output_dir = 'train'
num_files = 10000

with open(input_file, 'r') as infile:
    for i, line in enumerate(infile):
        if i >= num_files:
            break
        try:
            # Validate JSON
            json_object = json.loads(line)
            output_filename = os.path.join(output_dir, f"{i}.json")
            with open(output_filename, 'w') as outfile:
                # Write formatted JSON
                json.dump(json_object, outfile, indent=4)
        except json.JSONDecodeError as e:
            print(f"Skipping line {i+1} due to JSON decode error: {e}")
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} files...")

print(f"Successfully split the first {num_files} lines into individual JSON files in '{output_dir}'.") 