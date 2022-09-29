# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields, load_stoplist
import os
import spacy
import numpy as np

system = "wcph113"

my_language_model_de = language_model_path(system)

print(my_language_model_de)

vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" )

lists_path = vocab_lists_dicts_directory(system)
list_of_keywords=["Flammen", "Schwelbrand", "Explosion", "brennen"]

nlp = spacy.load(my_language_model_de)
all_output_words, all_vectors = [], []

for word in list_of_keywords:
    try:
        ms = nlp.vocab.vectors.most_similar(
            np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=2)
    except KeyError:
        continue

    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    #vectors = [nlp.vocab.get_vector(w) for w in ms[0][0]]
    all_output_words.extend(words)
    all_output_words.extend(list_of_keywords)
    compl_words = ["Ruhe"]
    all_output_words.extend(compl_words)

    #vocab_list = load_stoplist(vocabulary_path)
    #all_output_words_reduced = [word for word in all_output_words if word in vocab_list]
    #all_words_string = " ".join(all_output_words)
    #doc = nlp(all_words_string)
    #lemma_list = [token.lemma_ for token in doc]

all_output_words = list(set(all_output_words))


for word in all_output_words:
    new_vector = nlp.vocab.get_vector(word)
    all_vectors.append(new_vector)

print(all_output_words)
print(len(all_output_words))
print(len(all_vectors))


colors_list = ["red" if word in compl_words else "blue" if word in list_of_keywords else "green" for word in all_output_words]
print(colors_list)
vectors = np.stack(all_vectors, axis=0)
print(vectors)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
vecs = pca.fit_transform(vectors)

plt.scatter(vecs[:,0], vecs[:,1], c=colors_list)
for i, name in enumerate(all_output_words):
    plt.annotate(name, vecs[i])
plt.show()


fig = plt.figure(figsize=[5,5])
ax = plt.axes(projection="3d")

pca = PCA(n_components=3)
vecs = pca.fit_transform(vectors)


for i, name in enumerate(all_output_words):

    ax.scatter3D(vecs[i, 0], vecs[i, 1], vecs[i, 2], c=colors_list[i])
    ax.text(vecs[i, 0], vecs[i, 1], vecs[i, 2], '%s' % (str(name)), size=10, zorder=1,
            color=colors_list[i])

plt.title("Dimensionsreduktion für nächste Vektoren")
plt.show()

#semantic_words_list = keywords_to_semantic_fields(list_of_keywords=["erschrecken"], n_most_relevant_words=50,
 #                                         spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))


#print(semantic_words_list)