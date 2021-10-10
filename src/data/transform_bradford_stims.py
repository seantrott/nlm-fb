"""Add more examples to Bradford et al (2020) stims, controlling for first mention.

Currently, each item confounds "initial location" (which is also "false location") with "first mention".

We know that some of the NLMs tested seem to have a bias for thinking the first-mentioned referent 
in a passage is more likely to be mentioned again. So, it's important to deconfound first-mention
from true/false location.

Here, we do that, by simply adding a brief clause/sentence to each stimulus that mentions the alternative location.
"""

import pandas as pd

import spacy


## TODO: Consider modifying most recent mention too?

## Import spacy to idnetify noun chunks
nlp = spacy.load("en_core_web_sm")

## Edge case mappings: where spacy gets it wrong
mappings = {
	'Mum baked cakes': 'Mum',
	'the CD': 'Emily'
}

## Read in stims
df_passages = pd.read_csv("data/stims/bradford-fb.csv")

## Keep track of rows
all_rows = []
for index, row in df_passages.iterrows():

	row_dict = dict(row)
	row_dict['Modified'] = 'No'
	row_dict['First_mention'] = 'Start'
	row_dict['Recent_mention'] = 'End'
	row_dict['prior_mentions_start'] = 1
	row_dict['prior_mentions_end'] = 1
	row_dict['num_sentences'] = 3
	all_rows.append(row_dict)

	scenario = row['Scenario']
	
	first_sentence_nouns = list(nlp(scenario).noun_chunks)
	first_noun = first_sentence_nouns[0]

	## Edge cases
	if len(first_noun) > 1 and first_noun.text in mappings.keys():
		first_noun = mappings[first_noun.text]
	# name = scenario.split()[0]

	## Initial location of object (in FB, this is where the person thinks it is).
	first_location = scenario.split(".")[0].split()[-1]
	## Second location of object (where object gets moved).
	true_location = scenario.split(".")[1].split()[-1]
	## Final location (i.e., where the person looks).
	final_location = scenario.split(".")[2].split()[-1]

	## Craft a new sentence/clause with "End" mentioned first.
	new_sentence = "{entity} passed by the {tl}".format(entity = first_noun, tl = true_location)

	## Create new passage with this sentence in it.
	new_passage = "{ns}. {rest_of_passage}".format(ns = new_sentence, rest_of_passage=scenario)

	new_row_dict = row_dict.copy()
	new_row_dict['Modified'] = 'Yes'
	new_row_dict['Scenario'] = new_passage
	new_row_dict['First_mention'] = 'End'
	new_row_dict['Recent_mention'] = 'End'
	new_row_dict['prior_mentions_start'] = 1
	new_row_dict['prior_mentions_end'] = 2
	new_row_dict['num_sentences'] = 4
	all_rows.append(new_row_dict)

	## Craft a new sentence/clause with "Start" mentioned first.
	new_sentence = "{entity} passed by the {fl}".format(entity = first_noun, fl = first_location)

	## Create new passage with this sentence in it.
	new_passage = "{ns}. {rest_of_passage}".format(ns = new_sentence, rest_of_passage=scenario)

	new_row_dict = row_dict.copy()
	new_row_dict['Modified'] = 'Yes'
	new_row_dict['Scenario'] = new_passage
	new_row_dict['First_mention'] = 'Start'
	new_row_dict['Recent_mention'] = 'End'
	new_row_dict['prior_mentions_start'] = 2
	new_row_dict['prior_mentions_end'] = 1
	new_row_dict['num_sentences'] = 4
	all_rows.append(new_row_dict)


	####### Also create some passages with recent_mention == start (and also recent_mention == end)
	passage_sentences = scenario[0:-1].split(". ")[0:3]

	## recent_mention == start
	additional_sentence = "{entity} was near the {fl}".format(entity = first_noun, fl = first_location)
	sentences_copy = passage_sentences.copy()
	sentences_copy.insert(2, additional_sentence)
	new_passage = ". ".join(sentences_copy) + "."
	new_row_dict = row_dict.copy()
	new_row_dict['Modified'] = 'Yes'
	new_row_dict['Scenario'] = new_passage
	new_row_dict['First_mention'] = 'Start'
	new_row_dict['Recent_mention'] = 'Start'
	new_row_dict['prior_mentions_start'] = 2
	new_row_dict['prior_mentions_end'] = 1
	new_row_dict['num_sentences'] = 4
	all_rows.append(new_row_dict)

	additional_sentence = "{entity} was near the {tl}".format(entity = first_noun, tl = true_location)
	sentences_copy = passage_sentences.copy()
	sentences_copy.insert(2, additional_sentence)
	new_passage = ". ".join(sentences_copy) + "."
	new_row_dict = row_dict.copy()
	new_row_dict['Modified'] = 'Yes'
	new_row_dict['Scenario'] = new_passage
	new_row_dict['First_mention'] = 'Start'
	new_row_dict['Recent_mention'] = 'End'
	new_row_dict['prior_mentions_start'] = 1
	new_row_dict['prior_mentions_end'] = 2
	new_row_dict['num_sentences'] = 4
	all_rows.append(new_row_dict)



df_new_passages = pd.DataFrame(all_rows)
print(len(df_new_passages))
df_new_passages.to_csv("data/stims/bradford-fb-modified.csv")


