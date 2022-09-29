system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, heftroman_base_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
import os
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.linear_model import LinearRegression


scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}

dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
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

print(df.max_danger_typ.values)

df = df.rename(columns=columns_transl_dict)

df["genre"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "genre"], axis=1)
df["Reihe"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "series"], axis=1)

df["Reihe"] = df["Reihe"].apply(lambda x: "Julia" if "Julia" in x else x)


df = df[df["genre"] == "liebe"]



dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))




reihen = list(set(df.Reihe.values.tolist()))
reihen_colors = ["magenta", "red", "purple", "pink", "blue", "brown", "black", "orange", "cyan"]
reihen_dict = dict(zip(reihen, reihen_colors[:len(reihen)]))

reihen_mpatches_list = []
for reihe, color in reihen_dict.items():
    patch = mpatches.Patch(color=color, label=reihe)
    reihen_mpatches_list.append(patch)

y_variable = "Angstempfinden" # "Liebe" "Angstempfinden

values_dict = {}
for reihe in reihen:
    reihe_df = df[df["Reihe"]==reihe]
    print(reihe_df["max_danger_typ"].values.tolist())

    max_danger_colors = [dangers_dict[k] for k in reihe_df["max_danger_typ"].values.tolist()]

    fig, ax = plt.subplots()
    plt.scatter(reihe_df.loc[:, "max_value"], reihe_df.loc[:, y_variable], color=max_danger_colors)
    # regr = LinearRegression()
    # regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    # y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    # plt.plot(df.loc[:, x_variable], y_pred, color="green", linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=dangers_mpatches_list)

    plt.ylabel(y_variable)
    plt.xlabel("Gefahrenwert des Textabschnitts")
    plt.title("Liebesromane: Gefahrentyp und Angst-Korrelation für Reihe: " + reihe.capitalize())
    plt.show()

    values_dict[reihe] = [mean(reihe_df.loc[:, "max_value"]), mean(reihe_df.loc[:, "Angstempfinden"])
                           ]

print(values_dict)
values_df = pd.DataFrame.from_dict(values_dict, orient="index")
values_df.rename(index= {"Baccara": 2020, "Julia": 2000, "VERGISS MEIN NICHT": 1920}, inplace=True)
values_df.rename(columns={0:"Gefahrenlevel", 1: "Angstempfinden"}, inplace=True)
values_df.sort_index(inplace=True)

#values_df = values_df.set_index("year")
print(values_df)
fig, ax = plt.subplots()
values_df.plot(kind="line", color=["black", "grey"])
plt.title("Zeitlicher Verlauf: Gefahr und Angst in Liebesromanen")
plt.ylabel("Gefahren- und Angstlevel")
plt.xticks(rotation=45)
plt.show()

#fix, ax = plt.subplots()
values_df.plot(kind="bar" ,  color=["black", "grey"])
plt.ylabel("Gefahren- und Angstlevel")
plt.title("Zeitlicher Verlauf: Gefahr und Angst in Liebesromanen")
plt.xticks(rotation=45)
plt.show()

fig, ax = plt.subplots()
for reihe in reihen:
    reihe_df = df[df["Reihe"]==reihe]

    reihen_color = reihen_dict[reihe]
    x_variable = "max_value"


    plt.scatter(reihe_df.loc[:, x_variable], reihe_df.loc[:, y_variable], color=reihen_color)
    regr = LinearRegression()
    regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df.loc[:, x_variable], y_pred, color="black", linewidth=3)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=reihen_mpatches_list)

plt.ylabel(y_variable)
plt.xlabel("Gefahrenwert des Textabschnitts")
plt.title("Liebesromane: Gefahr-Angst-Korrelation")
plt.show()