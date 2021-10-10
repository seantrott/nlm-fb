"""Run BERT on passage data."""

import math
import numpy as np
import pandas as pd
import re

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



def main(filename, c1, c2):
    """
    Run BERT on stims. 

    Requires that filename point to a .csv file with a "passage" column, as well as 
    columns for the two words (c1 vs. c2) to be comparing (e.g., start vs. end).

    (#TODO: This main function could/should probabably be generalized to encompass other models too.)
    """

    # Assemble data path
    data_path = "data/stims/{FILE}.csv".format(FILE=filename)
    
    # Read in stims
    df_passages = pd.read_csv(data_path)
    print(len(df_passages))

    # For each passage, get log-odds of c1 vs. c2
    odds = []
    with tqdm(total=df_passages.shape[0]) as pbar:            
        for index, row in df_passages.iterrows():

            # Get passage
            text = row['passage']
            # Get substitutions for masked word
            candidates = [row[c1], row[c2]]
            # Get probabilities for these words
            p = mask_probability(text, candidates)
            # Get log odds of these probabilities (of c1 vs. c2)
            ratio = math.log(p[row[c1]]/p[row[c2]])
            # Add ratio to log-odds list
            odds.append(ratio)

            pbar.update(1)

    # Add to dataframe
    df_passages['log_odds'] = odds

    # Save file
    df_passages.to_csv("data/processed/{TASK}_bert-large_surprisals.csv".format(TASK=filename))


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, dest="filename",
                        default="fb")
    parser.add_argument("--c1", type=str, dest="c1",
                        default="start")
    parser.add_argument("--c2", type=str, dest="c2",
                        default="end")
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)