system = "wcph113"

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from ast import literal_eval
from collections import Counter
from sklearn.neighbors import NearestCentroid
import numpy as np
from math import dist

from preprocessing.presetting import local_temp_directory, heftroman_base_directory

data_df = pd.read_csv(os.path.join(local_temp_directory(system), "MaxDangerCharacters.csv"), index_col=0)

meta_df = pd.read_csv(os.path.join(heftroman_base_directory(system), "meta.tsv"), sep="\t").set_index("id")

print(meta_df)
print(data_df)


df = pd.concat([data_df, meta_df] ,axis=1)


y_variable =  "centr_EndChar" # "symp_EndChar" #  "Angstempfinden" #
x_variables = ["max_value"] #["centr_EndChar"] #

df = df.dropna(subset=[y_variable])
df = df.dropna(subset=[x_variables[0]])

print(df)

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi", "krieg", "western", "fantasy"]
colors_list = ["green", "black", "red", "purple", "cyan", "brown", "orange", "magenta"]
zipped_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

df = df[df.isin({"genre": genres_list}).any(1)]

genre_name = "alle Gattungen" # "liebe" #"krieg" #"krimi" # "horror"

if genre_name == "krimi":
    genre_name_legend = "Krimi"
elif genre_name == "krieg":
    genre_name_legend = "Kriegsroman"
elif genre_name == "liebe":
    genre_name_legend = "Liebesroman"
else:
    genre_name_legend = genre_name

#df_k = df[df["genre"] == genre_name]
df_k = df
print(df_k)
gender_list = df_k.gender_EndChar.values.tolist()
gender_counter = Counter(gender_list)
print("Gender Counter: ", gender_counter)


gender_colors = ["red" if x == "female" else "blue" if x == "male" else "green" for x in gender_list ]



prot_list = df_k.EndChar_series_protagonist.values.tolist()
print(prot_list)
prot_colors = ["black" if x == True else "cyan" for x in prot_list]
print(prot_colors)
for x_variable in x_variables:

    if x_variable == "max_value":
        x_variable_legend = "Gefahrenlevel"
    elif x_variable == "centr_EndChar":
        x_variable_legend = "Zentralität (degree) der gefährdeten Figur"
    else:
        x_variable_legend = x_variable
    if y_variable == "symp_EndChar":
        y_variable_legend = "Sympathiepotenzial der gefährdeten Figur"
    elif y_variable == "weigh_centr_EndChar":
        y_variable_legend = "Gewichtete Zentralität (degree) der gefährdeten Figur"
    elif y_variable == "centr_EndChar":
        y_variable_legend = "Zentralität (degree) der gefährdeten Figur"
    else:
        y_variable_legend = y_variable

    plt.scatter(df_k.loc[:, x_variable], df_k.loc[:, y_variable], color=gender_colors)
    regr = LinearRegression()
    regr.fit(df_k.loc[:, x_variable].array.reshape(-1, 1), df_k.loc[:, y_variable])
    y_pred = regr.predict(df_k.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df_k.loc[:, x_variable], y_pred, color="green", linewidth=3)
    plt.ylabel(y_variable_legend)
    plt.xlabel(x_variable_legend)
    plt.title("Korrelation für "+ genre_name_legend + " (für gefährlichsten Abschnitt jedes Text)")
    plt.show()

    fig, ax = plt.subplots()
    mpatches_list = []
    for genre, color in zipped_dict.items():
        values_dict = {"genre": [genre]}
        genre_df = df[df.isin(values_dict).any(1)]
        print(genre_df)

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

    plt.xlabel(x_variable_legend)
    plt.ylabel(y_variable_legend)
    plt.title("Korrelation für jeweils intensivsten Abschnitt nach Gattungen")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles= mpatches_list)
    plt.show()

    fig, ax = plt.subplots()
    clf = NearestCentroid()
    clf.fit(df[[x_variable, y_variable]].values, df["genre"])
    centroids = clf.centroids_
    print(centroids)
    class_labels = clf.classes_.tolist()
    print(class_labels)

    new_colors_list = [zipped_dict[k] for k in class_labels if k in colors_list]
    ax.scatter(centroids[:,0], centroids[:,1], color=new_colors_list)

    print(df[df["genre"] == "fantasy"])
    print(class_labels.index("fantasy"))
    print(centroids[class_labels.index("fantasy")])

    df["dist_centroid"] = df.apply(lambda x: dist([x[x_variable], x[y_variable]], [centroids[class_labels.index(x["genre"])][0], centroids[class_labels.index(x["genre"])][1]]), axis=1  )

    print(df.dist_centroid)

    for i in range(len(class_labels)):
        class_label = class_labels[i]

        if class_label in genres_list:
            print(class_label)
            class_df = df[df["genre"] == class_label]

            class_av_dist = class_df["dist_centroid"].mean()
            print(class_av_dist)

            xy = (centroids[i])
            print(xy)

            color = zipped_dict[class_label]
            print(color)
            circle = plt.Circle(xy, class_av_dist, color= color, fill=False)
            ax.add_patch(circle)

            plt.annotate(text = str(class_labels[i]), xy= xy , color= color)



    plt.xlabel(x_variable_legend)
    plt.ylabel(y_variable_legend)
    #plt.xlim(0, 0.8)
    #plt.ylim(0.3, 0.8)
    plt.title("Ausbreitung und Position nach Genres")
    plt.show()
