system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')


from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory, language_model_path, vocab_lists_dicts_directory
from preprocessing.corpus import DocNetworkfeature_Matrix

import os



my_model_de = language_model_path(system)

corpus_path = os.path.join(local_temp_directory(system), "Heftromane_texte_PER")
outfile_path_df = os.path.join(global_corpus_representation_directory(system), "Heftromane_SNA_Matrix.csv")
outfile_path_characters_list = os.path.join(global_corpus_representation_directory(system), "corpus_characters_list.txt")

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")

network_matrix_object = DocNetworkfeature_Matrix(corpus_path=corpus_path, language_model=my_model_de, segmentation_type="relative",
                                                 lemmatize=False, remove_hyphen=False, correct_ocr=False, normalize_orthogr=False,
                                                 normalization_table_path=normalization_table_path,
                                                 reduce_to_words_from_list=False, reduction_word_list=None,
                                                 )
network_matrix_object.generate_df()

network_matrix_object.save_csv(outfile_path= outfile_path_df)
print(network_matrix_object.corpus_characters_list)
network_matrix_object.corpus_characters_list_to_file(outfilepath=outfile_path_characters_list)