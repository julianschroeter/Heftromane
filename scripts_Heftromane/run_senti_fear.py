system = "wcph113"

if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import pandas as pd
import os
import spacy

from preprocessing.presetting import local_temp_directory, vocab_lists_dicts_directory, language_model_path, save_stoplist

emo_df = pd.read_csv(os.path.join(vocab_lists_dicts_directory(system), "emotion_lexicon.csv"))

print(emo_df)

fear_df = emo_df[emo_df["Fear"] == 1]

fear_words_list = fear_df["German (de)"].values.tolist()
fear_words = " ".join(fear_words_list)
print(fear_words)

model = language_model_path(system)
print(model)
nlp = spacy.load(model)
doc =nlp(fear_words)

fear_adjs = [word.lemma_ for word in doc if word.pos_ == "ADJ"]
print(fear_adjs)

print("Panik" in fear_words_list)
print( "panisch" in fear_adjs)

filename = "ResultList_SentiFear.txt"
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(fear_words_list, wordlist_filepath)