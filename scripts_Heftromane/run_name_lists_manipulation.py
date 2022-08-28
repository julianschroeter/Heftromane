system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')


from preprocessing.presetting import local_temp_directory, language_model_path, save_stoplist, vocab_lists_dicts_directory
import os
from preprocessing.presetting import merge_several_stopfiles_to_list, load_stoplist, save_stoplist

corpus_names_path = os.path.join(local_temp_directory(system), "Heftromane_Corpus_CharNames.txt" )

corpus_names_list = load_stoplist(corpus_names_path)

prenames_dir = os.path.join(vocab_lists_dicts_directory(system), "lists_prenames")
namelists_paths = []

for filename in os.listdir(prenames_dir):
    namelists_paths.append(os.path.join(prenames_dir, filename))


merged_names_list = merge_several_stopfiles_to_list(namelists_paths)

merged_names_list = sorted(list(set(merged_names_list)))
print(merged_names_list)

new_corpus_names_list, probl_cases = [], []
for name in corpus_names_list:
    if name in merged_names_list:
        new_corpus_names_list.append(name)
    elif name not in merged_names_list:
        probl_cases.append(name)

print("updated corpus names: ", new_corpus_names_list)

print("problematic cases: ", probl_cases)

new_corpus_names_path = os.path.join(prenames_dir, "new_Heftromane_Names.txt")
save_stoplist(new_corpus_names_list, new_corpus_names_path)

probl_cases_path = os.path.join(prenames_dir, "probl_cases_names.txt")
save_stoplist(probl_cases, probl_cases_path)



