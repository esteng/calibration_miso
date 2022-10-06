import re
import sys


path_to_target = sys.argv[1]
broken_apos_gex = re.compile(r"\\u2019")

lines_to_write = []
with open(path_to_target, 'r') as f:
    for line in f:
        new_line = broken_apos_gex.sub("'", line)
        if new_line != line: 
            print(new_line)
            print(line)
            print()
            line = new_line
        lines_to_write.append(line)

with open(path_to_target, 'w') as f:
    f.writelines(lines_to_write)