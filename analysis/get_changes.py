import pathlib
from collections import defaultdict

incorrect_sets = {}
correct_sets = {}
dirs = ["5000_100", "10000_100", "20000_100", "50000_100", "100000_100"]
for dir in dirs:
    correct_set = [x.name for x in pathlib.Path(dir).joinpath("correct").glob("*")]
    incorrect_set = [x.name for x in pathlib.Path(dir).joinpath("incorrect").glob("*")]

    correct_sets[dir] = set(correct_set)
    incorrect_sets[dir] = set(incorrect_set)

changed = []

for i in range(1, len(dirs)):
    curr_cset = correct_sets[dirs[i]]
    prev_cset = correct_sets[dirs[i-1]]
    curr_iset = incorrect_sets[dirs[i]]
    prev_iset = incorrect_sets[dirs[i-1]]

    incorrect_to_correct = ", ".join(list(prev_iset & curr_cset))
    correct_to_incorrect = ", ".join(list(prev_cset & curr_iset))

    total_change = ", ".join(list(curr_cset ^ prev_cset))
    for p in curr_cset ^ prev_cset:
        changed.append(p)

    perc = len(curr_cset ^ prev_cset) / len(curr_cset | curr_iset)
    cperc = len(prev_iset & curr_cset) / len(prev_iset)
    iperc = len(prev_cset & curr_iset) / len(prev_cset)

    print(f"from {dirs[i-1]} to {dirs[i]}")
    print(f"the following changed: {total_change}, \n\twhich is {perc:.2f} of total programs")
    print(f"became correct: {incorrect_to_correct}, \n\twhich is {cperc:.2f} of incorrect programs")
    print(f"became incorrect: {correct_to_incorrect}, \n\twhich is {iperc:.2f} of correct programs")
    print() 


print(f"total changes: {len(changed)}, unique changes {len(set(changed))}") 
change_count = defaultdict(int)
for c in changed:
    change_count[c]+=1
print(f"repeat offenders: {[(k,v) for k,v in change_count.items() if v > 1]}")
