"""Code to generate variations on different passage templates."""

import pandas as pd
import random


## TODO: Comprehension checks for BERT (e.g., {the {X} is on the ...})

## Read in templates
df_templates = pd.read_csv("data/templates/fb_templates.csv")

## Substitutions (TODO: Finalize/discuss these)
ITEMS = ['book',
        'magazine']

## TODO: Get more names and different genders!
NAMES = ['Antonio', 'Sean', 'James', 'Cameron', 'Tyler', 'Ben',
         'Sonja', 'Pamela', 'Emily', 'Catherine', 'Lisa', 'Marta']

LOCATION_PAIRS = [
    ('bed', 'table'),
    ('table', 'bed'),
    ('table', 'desk'),
    ('desk', 'table'),
    ('desk', 'bed'),
    ('bed', 'desk')
]

## Generate and save FB passages


all_passages = []
for index, row in df_templates.iterrows():
    for i in ITEMS:
        ## TODO: Check this --> do we want to just sample a few names?
        ## This isn't horrible --> we'll still sample from the whole set of names, but 
        ## we'll just only have a few *per* cell.
        for a in random.sample(NAMES, 2):
            ## Just get a few names for {B}
            for b in random.sample([i for i in NAMES if i != a], 2):
                ## Skip cases where names would be the same
                template = dict(row)
                if a == b:
                    continue
                for lp in LOCATION_PAIRS:
                    passage_template = template['passage']
                    passage = passage_template.format(
                        A = a, X = i, B = b, start = lp[0], end = lp[1]
                    )
                    all_passages.append({
                        'item': i,
                        'a': a,
                        'b': b,
                        'start': lp[0],
                        'end': lp[1],
                        'passage': passage,
                        'condition': template['condition'],
                        'first_mention': template['first_mention'],
                        'mentions_start': template['mentions_start'],
                        'mentions_end': template['mentions_end'],
                        'stim_number': template['item'],
                        'knowledge_cue': template['knowledge_cue']
                    })


df_passages = pd.DataFrame(all_passages)
print(len(df_passages))

df_passages.to_csv("data/stims/fb.csv")
