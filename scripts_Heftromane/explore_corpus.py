system = "wcph113"

from preprocessing.presetting import global_corpus_raw_dtm_directory, heftroman_corpus_directory, heftroman_base_directory, vocab_lists_dicts_directory, load_stoplist
from preprocessing.corpus import DTM
from clustering.my_pca import PC_df



import os
import pandas as pd

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))

dtm_filename = "heftromane_sraw_dtm_l1__use_idf_False2500mfw.csv"
dtm_path = os.path.join(global_corpus_raw_dtm_directory(system), dtm_filename)

metadata_filepath = os.path.join(heftroman_base_directory(system), "meta.tsv" )

metadata_df = pd.read_csv(metadata_filepath, sep="\t")
metadata_df = metadata_df.set_index("id")
print(metadata_df)

dtm_obj = DTM(data_matrix_filepath=dtm_path, metadata_csv_filepath=metadata_filepath)
dtm_obj = dtm_obj.add_metadata(["genre"])

print(dtm_obj.data_matrix_df)

pc_df = PC_df(input_df=dtm_obj.data_matrix_df)
pc_df.generate_pc_df()
pc_df.scatter(colors_list)

print(pc_df.component_loading_df.iloc[0, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[1, :].sort_values(ascending=False)[:20])
print(pc_df.component_loading_df.iloc[0,:].sort_values(ascending=True)[:20])
print(pc_df.component_loading_df.loc[1,: ].sort_values(ascending=True)[:20])
print(pc_df.pca.explained_variance_)