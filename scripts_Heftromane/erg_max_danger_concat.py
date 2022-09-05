system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, heftroman_base_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from topicmodeling.postprocessing import ChunksFeatureMatrix
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ast import literal_eval
import numpy as np


def replace_full_name(name, list_of_names):
    try:
        matching = [s for s in list_of_names if name in s][0]
    except:
        matching = name
    return matching

scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}


infile_path = os.path.join(local_temp_directory(system), "erg_DocSuspenseFormsMatrix.csv")
metadata_filepath = os.path.join(local_temp_directory(system), "erg_meta.tsv" )

sent_fear_path = os.path.join(local_temp_directory(system), "erg_DocSentFearMatrix.csv")

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)
df = matrix.data_matrix_df

sent_fear_df = pd.read_csv(sent_fear_path, index_col=0)
df = pd.concat([df, sent_fear_df], axis=1)

df = df.rename(columns=columns_transl_dict)
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


danger_df = df.drop(columns=["Liebe", "SentLexFear", "embedding_Angstempfinden","UnbekannteEindr", "Angstempfinden", "großzügig", "verwerflich"])

danger_df["max_value"] = danger_df.max(axis=1)
danger_df["max_danger_typ"] = danger_df.idxmax(axis=1)
danger_df["embedding_Angstempfinden"] = df["embedding_Angstempfinden"]
danger_df["Angstempfinden"] = df["Angstempfinden"]
danger_df["UnbekannteEindr"] = df["UnbekannteEindr"]
danger_df["Liebe"] = df["Liebe"]
danger_df["doc_chunk_id"] = danger_df.index
#danger_df["doc_chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-4])
danger_df["doc_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[:-5])
danger_df["chunk_id"] = danger_df["doc_chunk_id"].apply(lambda x: str(x)[-4:])

print(danger_df)




endanger_char_df = pd.read_csv(os.path.join(local_temp_directory(system), "erg_DocNamesCounterMatrix.csv"), index_col=0)

print(endanger_char_df)

max_danger_character_df = pd.concat([danger_df, endanger_char_df], axis=1)
max_danger_character_df.rename(columns={"0":"EndCharName"}, inplace=True)

max_danger_character_df.set_index("doc_id", inplace=True)

print(max_danger_character_df)

max_danger_character_df.drop_duplicates(inplace=True)


max_danger_character_df = max_danger_character_df[~max_danger_character_df.index.duplicated(keep='first')]



danger_df.to_csv(os.path.join(local_temp_directory(system), "erg_AllChunksDangerCharacters.csv"))