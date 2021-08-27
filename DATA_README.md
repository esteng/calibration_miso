
# Data

## Organization

For each function, there are varying splits across total number of training examples and total number of function examples (e.g. 5000_100 = 5000 training examples, 100 of which are the function). 
Each function will have it's own dir, with the splits as subdirs. Each subdir will have train, test, and dev files. 

## File types

- `XXX.src_tok`: tokenized user utterances with previous user utterance and agent utterance included, separated by special tokens (`__User` and `__Agent`) 
- `XXX.tgt`: tokenized linearized lispress sequences 
- `XXX.idx`: file of line indices to use in lookup later.
- (optional) `generated_lookup_table.json`: This is the lookup table for minimal pairs. It is a dict where the keys are line idxs, and the values are lists of line idxs. The keys cover the training set of FindManager examples, and each example is paired with a ranked list of indices in the train set which correspond to their minimal pairs. For example, if the entry is `{3: [1, 8, 7]}` then train line at index 3 contains FindManager, and is most similar to the train line at position 1, which does not contain FindManager in the output. 
- (optional) `lookup_table.json`: For manually annotated minimal pairs, this lookup table has line number keys and values of `(source, target)` strings, where the source and target are the generated minimal pair.

## Minimal pair extraction
To extract minimal pairs for a given function, the following process is run: 
1. Find all examples in train containing that function
2. Get the subset of the training data that does not contain the function 
3. For each example from step 1, rank each example from step 2.
	- Ranking is done based on Levenshtein distance between the user utterances. 
4. Return top k examples from step 3

The reason we use top k rather than top 1 here is that some examples may have the same closest pair, so there should be a ranked list of candidates in case the top candidate has already been taken. 

