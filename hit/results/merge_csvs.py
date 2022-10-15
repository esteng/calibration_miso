import csv
import pathlib
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    all_data = []
    for i, file in enumerate(results_dir.glob("Batch*.csv")): 
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
            all_data += data 
    with open(args.out_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
        writer.writeheader()
        for line in all_data:
            writer.writerow(line)