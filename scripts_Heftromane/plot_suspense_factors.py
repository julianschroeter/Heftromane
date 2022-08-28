system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')

import pandas as pd
from preprocessing.presetting import global_corpus_representation_directory, heftroman_base_directory
from preprocessing.corpus import DocFeatureMatrix
from topicmodeling.postprocessing import ChunksFeatureMatrix
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}


infile_path = os.path.join(global_corpus_representation_directory(system), "DocSuspenseFormsMatrix.csv")
metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )


matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
print(df)

danger_df = df.drop(columns=["Liebe", "SentLexFear", "Angstempfinden", "UnbekannteEindr"])
print(danger_df)
danger_df["max_value"] = danger_df.max(axis=1)
danger_df["max_danger_typ"] = danger_df.idxmax(axis=1)
print(danger_df)

y_variable = "Angstempfinden"
x_variables = ["Gewaltverbrechen", "SentLexFear", "Kampf", "Krieg", "UnbekannteEindr", "Sturm", "Feuer", "Entführung", "Liebe"]



chunks_matrix = ChunksFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=metadata_filepath, metadata_df=None, mallet=False)
chunks_matrix.adjust_doc_chunk_multiindex()

means = chunks_matrix.last_chunk()
print(means.data_matrix_df)
means = means.add_metadata("genre")

genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi"]
colors_list = ["green", "black", "red", "yellow", "cyan"]
zipped_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

for x_variable in x_variables:
    plt.scatter(df.loc[:, x_variable], df.loc[:, y_variable], color="red")
    regr = LinearRegression()
    regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    plt.plot(df.loc[:, x_variable], y_pred, color="green", linewidth=3)
    plt.ylabel(y_variable)
    plt.xlabel(x_variable)
    plt.title("Korrelation (für Textende)")
    plt.show()

    fig, ax = plt.subplots()
    mpatches_list = []
    for genre, color in zipped_dict.items():
        genre_obj = means.reduce_to_categories("genre", [genre])
        genre_obj = genre_obj.eliminate(["genre"])
        genre_df = genre_obj.data_matrix_df
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
    plt.title("Korrelation (für Textende nach Gattungen)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles= mpatches_list)
    plt.show()