system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/git/Heftromane')


from preprocessing.presetting import local_temp_directory, language_model_path, save_stoplist

import os



my_model_de = language_model_path(system)

corpus_path = os.path.join(local_temp_directory(system), "Heftromane_texte_PER")
corpus_names_path = os.path.join(local_temp_directory(system), "Heftromane_Corpus_CharNames.txt" )

searched_name = "Helling"

all_corpus_names = []

for path, dir, filenames in os.walk(corpus_path):
    for filename in filenames:
        position = None
        text = open(os.path.join(path, filename), "r").read()
        names_list = text.split(" ")

        names = list(set(names_list))
        all_corpus_names.extend(names)

        count = 0
        for name in names_list:
            if searched_name not in name:
                count += 1
            else:
                position = count
                break
        if position is not None:
            rel_pos = position / len(names_list)
        else:
            rel_pos = None

        print("Position of Name: ", searched_name, position, "relative pos.: ", rel_pos)


all_corpus_names = sorted(list(set(all_corpus_names)))
print(all_corpus_names)

save_stoplist(all_corpus_names, corpus_names_path)
