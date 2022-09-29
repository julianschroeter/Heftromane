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
from statistics import mean, stdev
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def split_liste(liste, length = 5):
    nested_list = []
    for i in range(0, len(liste), length):
        sublist= liste[i:i+length]
        nested_list.append(sublist)
    return nested_list

def count_changes(liste):
    diffs = 0
    start_count = 0
    end_count = 1
    for i in range(len(liste)):
        while end_count < len(liste):
            first = liste[start_count]
            sec = liste[end_count]
            if first != sec:
                diffs += 1

            start_count += 1
            end_count += 1
    return diffs


def add_abs_diff(liste):
    diffs = 0
    start_count = 0
    end_count  = 1
    for i in range(len(liste)):
        while end_count < len(liste):
            first = liste[start_count]
            sec = liste[end_count]
            diff = abs(first - sec)
            diffs += diff
            start_count += 1
            end_count += 1
    return diffs





scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}

dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["dimgrey", "darkgrey", "black", "white", "lightgrey", "snow"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))

dangers_mpatches_list = []
for genre, color in dangers_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    dangers_mpatches_list.append(patch)


infile_path = os.path.join(local_temp_directory(system), "AllChunksDangerCharacters.csv")
metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )
metadata_df = pd.read_csv(metadata_filepath, sep="\t").set_index("id")

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)

df["genre"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "genre"], axis=1)

print(df)

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi", "krieg", "fantasy"]
genre_danger_dict = {}
for genre in genres_list:
    genre_df = df[df["genre"] == genre]
    genre_df = genre_df.sort_values(["doc_id", "doc_chunk_id"], ascending=[True, True])
    danger_types_list = genre_df.max_danger_typ.values.tolist()
    danger_values_list = genre_df.max_value.values.tolist()
    nested_danger_types_list = split_liste(danger_types_list)
    nested_danger_values_list = split_liste(danger_values_list)
    print(genre)
    print(nested_danger_types_list)

    danger_value_diffs = [add_abs_diff(sublist) for sublist in nested_danger_values_list]
    danger_typ_changes = [count_changes(sublist) for sublist in nested_danger_types_list]
    print(danger_typ_changes)
    print("value diffs, mean, std: ",mean(danger_value_diffs), stdev(danger_value_diffs))
    print("count changes, mean, std: ", mean(danger_typ_changes), stdev(danger_typ_changes))

    mean_level = [mean(sublist) for sublist in nested_danger_values_list]
    print("mean_danger_level: ", mean(mean_level))

    genre_danger_dict[genre] = [mean(danger_typ_changes), mean(danger_value_diffs)]
    fig, ax = plt.subplots()
    chunk_counts = [1,2,3,4,5]
    if genre == "krieg":
        sampled_types = nested_danger_types_list[50]
        sampled_values = nested_danger_values_list[50]
        max_danger_colors = [dangers_dict[k] for k in sampled_types]
        plt.bar(chunk_counts, sampled_values, color=max_danger_colors)
        plt.plot(chunk_counts, sampled_values, color="black")
        plt.title("Gefahrenlevel und Typ für exemplarischen Kriegsroman")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=dangers_mpatches_list)
        plt.show()
    if genre == "krimi":
        sampled_types = nested_danger_types_list[50]
        sampled_values = nested_danger_values_list[50]
        max_danger_colors = [dangers_dict[k] for k in sampled_types]
        plt.bar(chunk_counts, sampled_values, color=max_danger_colors)
        plt.plot(chunk_counts, sampled_values, color= "black")
        plt.title("Gefahrenlevel und Typ für exemplarischen Krimi")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=dangers_mpatches_list)
        plt.show()
    if genre == "abenteuer":
        sampled_types = nested_danger_types_list[50]
        sampled_values = nested_danger_values_list[50]
        max_danger_colors = [dangers_dict[k] for k in sampled_types]
        plt.bar(chunk_counts, sampled_values, color=max_danger_colors)
        plt.plot(chunk_counts, sampled_values, color="black")
        plt.title("Gefahrenlevel und Typ für exemplarischen Abenteuerroman")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=dangers_mpatches_list)
        plt.show()

print(genre_danger_dict)


genre_danger_df = pd.DataFrame.from_dict(genre_danger_dict, orient="index", columns=["Wechsel des Gefahrentyps", "Grad der Gefahrensteigerung"])

df = genre_danger_df

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=[genre.capitalize() for genre in df.index.tolist()], columns=df.columns)

df = df.sort_values(by=["Wechsel des Gefahrentyps"],ascending=True)


print(df)


#fig, ax = plt.subplots()
colors = ["black", "lightgrey"]
#plt.style.use('seaborn')
df.plot.bar(subplots=False, color = colors)
plt.xticks(rotation=30)
plt.title("Wechsel der Gefahrensituationen zwischen Textabschnitten")
plt.show()
df.plot.line()
plt.title("Wechsel der Gefahrensituationen zwischen Textabschnitten")
plt.xticks(rotation=45)
plt.show()