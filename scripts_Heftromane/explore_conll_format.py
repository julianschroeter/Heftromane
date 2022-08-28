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
print("k00300000128" in rhodan_ids)
cotton_ids = meta_df[meta_df["series"] == "Jerry Cotton"].index.values
sinclair_ids = meta_df[meta_df["series"] == "John Sinclair"].index.values
shark_ids = meta_df[meta_df["series"] == "TOM SHARK"].index.values

heftroman_stoplist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "heftroman_stopwords_header.txt"))

corpus_path = os.path.join(local_temp_directory(system), "conll_chunks_Heftromane")

outfile_dir_names = os.path.join(local_temp_directory(system), "Namen_Heftroman_chunks")
outfile_dir_semant = os.path.join(local_temp_directory(system), "SemantLemma_Heftroman_chunks")

if not os.path.exists(outfile_dir_names):
    os.makedirs(outfile_dir_names)
if not os.path.exists(outfile_dir_semant):
    os.makedirs(outfile_dir_semant)

corpus_names_path = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames", "rev_names_edjs.txt")
corpus_names_list = load_stoplist(corpus_names_path)
for filename in os.listdir(corpus_path):
    doc_basename = filename[:-5]
    print(doc_basename)

    if doc_basename == doc_basename:
        filepath = os.path.join(corpus_path, filename)
        df = pd.read_csv(filepath,engine="python" , sep=",", on_bad_lines="skip",index_col=0)
        print(df)

        df["Coref"] = df["Coref"].astype("category")
        df["PER"] = df["Coref"].apply(lambda x: True if "NER=PER" in x else False)

        pos_dict = {"POS": ["NN", "VVFIN", "VVPP", "VVINF", "ADJD", "ADJA"]}
        semant_df = df[df.isin(pos_dict).any(1)]
        names_df = df[df["PER"] == True]
        names_df = names_df[names_df["pars"] != "app"]

        semant_lemma_list = semant_df["Lemma"].values.tolist()
        semant_lemma_list = [str(word) for word in semant_lemma_list if word not in heftroman_stoplist]

        names_list = names_df["Lemma"].values.tolist()
        names_list = [str(word) for word in names_list if word not in heftroman_stoplist]
        names_list = [word for word in names_list if word in corpus_names_list]

        semant_text = " ".join(semant_lemma_list)
        names_text = " ".join(names_list)

        if doc_basename in rhodan_ids:
            names_text = names_text.replace("Perry Rhodan", "PerryRhodan")
            names_text = names_text.replace(" Rhodan", " PerryRhodan")
            names_text = names_text.replace("Perry ", "PerryRhodan ")
            names_text = names_text.replace("Percy Stuart", "PercyStuart")
            names_text = names_text.replace(" Stuart", " PercyStuart")
            names_text = names_text.replace("Percy ", "PercyStuart ")
            names_text = names_text.replace("Reginald Bull", "ReginaldBull")
            names_text = names_text.replace("Bully", "ReginaldBull")
            names_text = names_text.replace("Reginald ", "ReginaldBull ")
            names_text = names_text.replace(" Bull", " ReginaldBull")
            names_text = names_text.replace("Clark G. Flipper", "ClarkFlipper")
            names_text = names_text.replace("Clark Flipper", "ClarkFlipper")
            names_text = names_text.replace(" Flipper", " ClarkFlipper")
            print(names_text)
        elif doc_basename in cotton_ids:
            names_text = names_text.replace("Jerry Cotton", "JerryCotton")
            names_text = names_text.replace(" Cotton", " JerryCotton")
            names_text = names_text.replace("Jerry ", "JerryCotton ")

        elif doc_basename in sinclair_ids:
            names_text = names_text.replace("John Sinclair", "JohnSinclair")
            names_text = names_text.replace(" Sinclair", " JohnSinclair")
            names_text = names_text.replace("John ", "JohnSinclair ")

        elif doc_basename in shark_ids:
            names_text = names_text.replace("Tom Shark", "TomShark")
            print(names_text)
            names_text = names_text.replace("Tom ", "TomShark ")
            names_text = names_text.replace(" Shark", " TomShark")


        outfilepath = os.path.join(outfile_dir_names, filename)
        with open(outfilepath, "w") as f:
            f.write(names_text)

        outfilepath = os.path.join(outfile_dir_semant, filename)
        with open(outfilepath, "w") as f:
            f.write(semant_text)

    print("file with filename ", filename, "successfully processed")

print("process finished")

