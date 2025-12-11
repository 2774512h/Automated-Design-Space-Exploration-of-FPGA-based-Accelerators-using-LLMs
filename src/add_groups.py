import json
import argparse
import math
from typing import Dict, Any

def add_groups(
    input_path: str,
    output_path: str,
    group_size: int = 200,
):
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            rec  : Dict[str, Any] = json.loads(line)
            cid = int(rec["id"])
            group_id = cid // group_size
            rec["group_id"] = group_id
            fout.write(json.dumps(rec) + "\n")
            count += 1

    print(f"Processed {count} chunks. Group size = {group_size}.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-g", "--group_size", type=int, default=200)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    add_groups(args.input, args.output, args.group_size)

