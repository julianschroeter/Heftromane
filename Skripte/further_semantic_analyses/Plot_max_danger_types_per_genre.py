system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')


from preprocessing.presetting import heftroman_base_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
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

y_variable = "Angstempfinden"

dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))

dangers_mpatches_list = []
for genre, color in dangers_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    dangers_mpatches_list.append(patch)


infile_path = os.path.join(local_temp_directory(system), "MaxDangerCharacters.csv")
metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )
max_matrix = DocFeatureMatrix(data_matrix_filepath=infile_path, metadata_csv_filepath=metadata_filepath, metadata_df=None, mallet=False)
max_matrix = max_matrix.add_metadata(["genre"])

df = max_matrix.data_matrix_df

print(df)


genres_list = ["krimi", "horror", "liebe", "abenteuer", "scifi", "krieg", "fantasy"]
colors_list = ["green", "black", "red", "yellow", "cyan", "brown","orange"]
genres_dict = dict(zip(genres_list, colors_list[:len(colors_list)]))

genres_mpatches_list = []
for genre, color in genres_dict.items():
    patch = mpatches.Patch(color=color, label=genre)
    genres_mpatches_list.append(patch)

for genre in genres_list:
    genre_df = df[df["genre"] == genre]
    max_danger_colors = [dangers_dict[k] for k in genre_df["max_danger_typ"].values.tolist()]
    print(max_danger_colors)
    fig, ax = plt.subplots()
    plt.scatter(genre_df.loc[:, "max_value"], genre_df.loc[:, y_variable], color=max_danger_colors)
    #regr = LinearRegression()
    #regr.fit(df.loc[:, x_variable].array.reshape(-1, 1), df.loc[:, y_variable])
    #y_pred = regr.predict(df.loc[:, x_variable].array.reshape(-1, 1))
    #plt.plot(df.loc[:, x_variable], y_pred, color="green", linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=dangers_mpatches_list)

    plt.ylabel(y_variable)
    plt.xlabel("Gefahrenwert des gefährlichsten Textabschnitts")
    plt.title("Gefahrentyp und Angst-Korrelation für Gattung "+ genre.capitalize())
    plt.show()