system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import os
import pandas as pd
import numpy as np

from preprocessing.presetting import local_temp_directory,heftroman_base_directory, load_stoplist, vocab_lists_dicts_directory, word_translate_table_to_dict
from preprocessing.corpus import DTM


metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv")

meta_df = pd.read_csv(metadata_filepath, index_col=0, sep="\t")
print(meta_df)

rhodan_ids = meta_df[meta_df["series"] == "Perry Rhodan"].index.values
cotton_ids = meta_df[meta_df["series"] == "Jerry Cotton"].index.values
sinclair_ids = meta_df[meta_df["series"] == "John Sinclair"].index.values
shark_ids = meta_df[meta_df["series"] == "TOM SHARK"].index.values

heftroman_stoplist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "heftroman_stopwords_header.txt"))

corpus_path = os.path.join(heftroman_base_directory(system), "conll")
outfile_directory = os.path.join(local_temp_directory(system), "conll_chunks_Heftromane")

if not os.path.exists(outfile_directory):
    os.makedirs(outfile_directory)

corpus_names_path = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames", "rev_names_edjs.txt")
corpus_names_list = load_stoplist(corpus_names_path)
for filename in os.listdir(corpus_path):

    if filename == filename:
        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath,engine="python" , sep="\t", on_bad_lines="skip",names= ["nr", "Token", "Lemma", "?_", "POS", "Casus", "id", "pars", "_", "Coref"])

        dfs = np.array_split(df, 5)
        for i, chunk_df in enumerate(dfs):
            basename = os.path.basename(filepath)
            basename = os.path.splitext(basename)[0]
            chunk_filename = "{}{}{:04d}".format(basename, "_", i)
            chunk_df.to_csv(os.path.join(outfile_directory, chunk_filename))

print("process finished")