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
                       "Klinge":"Zweikampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe"}


infile_path = os.path.join(global_corpus_representation_directory(system), "DocSuspenseFormsMatrix.csv")
metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )


matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)
df_unscaled = df
scaled_features = scaler.fit_transform(df)
df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
print(df)

covar = df.cov()
print(covar)



y_variable = "Angstempfinden"
x_variables = ["Gewaltverbrechen", "SentLexFear", "Zweikampf", "Krieg", "UnbekannteEindr", "Sturm", "Feuer", "Entführung", "Liebe"]



chunks_matrix = ChunksFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=metadata_filepath, metadata_df=None, mallet=False)
chunks_matrix.adjust_doc_chunk_multiindex()

means = chunks_matrix.mean_doclevel()
means = means.add_metadata(["genre", "title", "author_norm"])
means_df = means.data_matrix_df

begin = chunks_matrix.first_chunk()
begin = begin.add_metadata(["genre", "title", "author_norm"])
begin_df = begin.data_matrix_df

end = chunks_matrix.last_chunk()
end = end.add_metadata(["genre", "title", "author_norm"])
end_df = end.data_matrix_df

