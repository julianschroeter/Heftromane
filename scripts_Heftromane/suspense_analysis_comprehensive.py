system = "wcph113"

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from ast import literal_eval
from collections import Counter

from preprocessing.presetting import local_temp_directory, heftroman_base_directory

data_df = pd.read_csv(os.path.join(local_temp_directory(system), "MaxDangerCharacters.csv"), index_col=0)

meta_df = pd.read_csv(os.path.join(heftroman_base_directory(system), "meta.tsv"), sep="\t").set_index("id")

print(meta_df)
print(data_df)


df = pd.concat([data_df, meta_df] ,axis=1)


y_variable = "Angstempfinden" #"symp_EndChar" #"centr_EndChar" #
x_variables = ["max_value"]

df = df.dropna(subset=[y_variable])
df = df.dropna(subset=[x_variables[0]])

print(df)

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi"]
colors_list = ["green", "black", "red", "yellow", "cyan"]
zipped_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

genre_name = "horror"
df_k = df[df["genre"] == genre_name]
print(df_k)
gender_list = df_k.gender_EndChar.values.tolist()
gender_counter = Counter(gender_list)
print("Gender Counter: ", gender_counter)
gender_colors = ["red" if x == "female" else "blue" if x == "male" else "white" for x in gender_list ]



prot_list = df_k.EndChar_series_protagonist.values.tolist()
print(prot_list)
prot_colors = ["black" if x == True else "cyan" for x in gender_list]
print(prot_colors)
for x_variable in x_variables:

    plt.scatter(df_k.loc[:, x_variable], df_k.loc[:, y_variable], color=gender_colors)
    regr = LinearRegression()
    regr.fit(df_k.loc[:, x_variable].array.reshape(-1, 1), df_k.loc[:, y_variable])
    y_pred = regr.predict(df_k.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df_k.loc[:, x_variable], y_pred, color="green", linewidth=3)
    plt.ylabel(y_variable)
    plt.xlabel(x_variable)
    plt.title("Korrelation für "+ genre_name + " (für gefährlichsten Abschnitt jedes Texts")
    plt.show()

    fig, ax = plt.subplots()
    mpatches_list = []
    for genre, color in zipped_dict.items():
        values_dict = {"genre": [genre]}
        genre_df = df[df.isin(values_dict).any(1)]


        plt.scatter(genre_df.loc[:,x_variable], genre_df.loc[:, y_variable], color=color, label=genre)

        regr = LinearRegression()
        regr.fit(genre_df.loc[:,x_variable].array.reshape(-1,1), genre_df.loc[:, y_variable])
        y_pred = regr.predict(genre_df.loc[:,x_variable].array.reshape(-1,1))
        plt.plot(genre_df.loc[:,x_variable], y_pred, color = color, linewidth=3)

        patch = mpatches.Patch(color=color, label=genre)
        mpatches_list.append(patch)
        #plt.show()
        #plt.scatter(krimis_df.loc[:,x_variable], krimis_df.loc[:, "Zweikampf"], color="green")
        #plt.show()

    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.title("Korrelation")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles= mpatches_list)
    plt.show()