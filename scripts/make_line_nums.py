# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path   
import sys

root_dir = Path(sys.argv[1])

for train_path in root_dir.glob("*.src_tok"):
    with open(train_path) as f1:
        line_num = len(f1.readlines())
    split = train_path.stem
    out_path = train_path.parent.joinpath(f"{split}.idx")
    with open(out_path,"w") as f1:
        for i in range(line_num):
            f1.write(f"{i}\n")


