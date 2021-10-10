"""Run BERT on passage data: just get surprisal for final word."""

import math
import numpy as np
import pandas as pd
import re

from collections import Counter
from nltk import word_tokenize
from tqdm import tqdm

from torch import tensor, softmax, no_grad
from transformers import (BertTokenizer, BertForMaskedLM,
                          # pipeline,
                          )

# Load pre-trained model tokenizer (vocabulary)
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased')

# Load pre-trained model (weights)
BERT = BertForMaskedLM.from_pretrained('bert-large-uncased')
BERT.eval()


def mask_probability(text, candidates, model=BERT, tokenizer=BERT_TOKENIZER):
    """Probabilities for candidates as replacement for [MASK] in text

    Args:
        text (str): Text containing a single [MASK]
        candidates (list of str): Candidate mask replacements
        model (TYPE, optional): language model (BERT)
        tokenizer (TYPE, optional): lm tokenizer (BERT)

    Returns:
        candidates (dict): {candidate: prob}
    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    # Get candidate ids
    candidate_ids = {}
    for candidate in candidates:
        candidate_tokens = candidate.split()
        candidate_ids[candidate] = tokenizer.convert_tokens_to_ids(
            candidate_tokens)

        # TODO: Check for 100 tokens

    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate, ids, in candidate_ids.items():

        # Add a mask for each token in candidate

        candidate_text = re.sub("\[MASK\] ", "[MASK] " * len(ids), text)

        # Tokenize text
        tokenized_text = tokenizer.tokenize(candidate_text)
        mask_inds = np.where(np.array(tokenized_text) == "[MASK]")

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = tensor([indexed_tokens])

        # Predict all tokens
        with no_grad():
            outputs = model(tokens_tensor)  # token_type_ids=segments_tensors)
            predictions = outputs[0]

        # get predicted tokens
        probs = []

        for (i, mask) in enumerate(mask_inds[0]):
            # prediction for mask
            mask_preds = predictions[0, mask.item()]
            mask_preds = softmax(mask_preds, 0)
            prob = mask_preds[ids[i]].item()
            probs.append(np.log(prob))

        candidate_probs[candidate] = np.exp(np.mean(probs))

    return candidate_probs


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s


def main(filename):
    """
    Run BERT on stims. Gets surprisal of final word in each stim.

    Requires that filename point to a .csv file with a "passage" column.
    """

    # Assemble data path
    data_path = "data/stims/{FILE}.csv".format(FILE=filename)
    
    # Read in stims
    df_passages = pd.read_csv(data_path)
    print(len(df_passages))

    # For each passage, get log-odds of c1 vs. c2
    probs = []
    beliefs = []
    consistency = []
    final_words = []
    masked_passages = []
    with tqdm(total=df_passages.shape[0]) as pbar:            
        for index, row in df_passages.iterrows():

            # conditions
            belief, con = row['Condition code'].split("-")
            beliefs.append(belief)
            consistency.append(con)

            # Get passage
            text = row['Scenario']
            # Get final word
            words = word_tokenize(text)
            final_word = words[-2]
            final_words.append(final_word)

            # Get number of occurrences
            num_occurrences = text.count(final_word)

            ## Mask final occurrence of word in passage
            masked_passage = nth_repl(text, final_word, "[MASK] ", num_occurrences)
            masked_passages.append(masked_passage)

            # Now look up probability for final word
            candidates = [final_word]
            # Get probabilities for these words
            p = mask_probability(masked_passage, candidates)
            
            # Add ratio to log-odds list
            probs.append(p[final_word])

            # Update progress bar
            pbar.update(1)

    # Add to dataframe
    df_passages['probability'] = probs
    df_passages['belief'] = beliefs
    df_passages['consistency'] = consistency
    df_passages['final_word'] = final_words
    df_passages['masked_passages'] = masked_passages

    # Save file
    print("Saving file")
    df_passages.to_csv("data/processed/{TASK}_bert-large_surprisals.csv".format(TASK=filename))


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, dest="filename",
                        default="bradford-fb")
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)