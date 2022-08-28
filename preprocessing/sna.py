import networkx as nx
import re
import spacy
from collections import Counter
from itertools import combinations

from preprocessing.text import Text



class NEnetwork(Text):
    def __init__(self,filepath, text, id, chunks, pos_triples, token_length, remove_hyphen, normalize_orthogr, normalization_table_path,
                 correct_ocr, eliminate_pagecounts, handle_special_characters, inverse_translate_umlaute,
                 eliminate_pos_items, keep_pos_items, list_keep_pos_tags, list_eliminate_pos_tags, lemmatize,
                 sz_to_ss,
                 reduce_to_words_from_list, reduction_word_list,
                 translate_umlaute, max_length,
                 remove_stopwords, stopword_list, language_model,
                 characters_list=None, characters_counter=None, character_pairs_counter=None, character_pairs_rel_freq=None,
                 graph=None, minimal_reference=2):
        Text.__init__(self, filepath, text, id, chunks, pos_triples, token_length, remove_hyphen, normalize_orthogr,
                      normalization_table_path,
                 correct_ocr, eliminate_pagecounts, handle_special_characters, inverse_translate_umlaute,
                 eliminate_pos_items, keep_pos_items, list_keep_pos_tags, list_eliminate_pos_tags, lemmatize,
                 sz_to_ss, translate_umlaute,
                 reduce_to_words_from_list, reduction_word_list,
                 max_length,
                 remove_stopwords, stopword_list, language_model)
        self.minimal_reference = minimal_reference
        self.characters_list = characters_list
        self.characters_counter = characters_counter
        self.character_pairs_counter = character_pairs_counter
        self.character_pairs_rel_freq = character_pairs_rel_freq
        self.graph = graph

    def generate_per_names_list(self, standardize=False, reduce_to_one_name=False):
        nlp = spacy.load(self.language_model)
        all_character_occurences = []
        final_characters_list = []
        doc = nlp(self.text[:self.max_length])
        per_names_list = [ent.names_text for ent in doc.ents if ent.label_ == "PER"]
        per_names_list = list(set(per_names_list))
        return per_names_list

    def generate_loc_names_list(self):
        nlp = spacy.load(self.language_model)
        all_character_occurences = []
        final_characters_list = []
        doc = nlp(self.text[:self.max_length])
        names_list = [ent.names_text for ent in doc.ents if ent.label_ == "LOC"]
        names_list = list(set(names_list))
        return names_list



    def generate_characters_graph(self, reduce_to_one_name):
        nlp = spacy.load(self.language_model)
        character_pairs_global =[]
        raw_all_character_occurences = []
        raw_nested_character_occurence = []
        final_characters_list = []
        if not self.chunks:
            print("Instance method self.f_chunking() has to be called first / self.chunks attribute must not be empty!")
            pass
        for chunk in self.chunks:

            paragr_characters = chunk.split(" ")




            raw_nested_character_occurence.append(paragr_characters)
            raw_all_character_occurences += paragr_characters

            paragr_characters_corr = []
            raw_one_word_names = []
            raw_composed_names = []





            # vereinheitliche zuerst Genitiv-s- und dann im nächsten Schritt altertümliche Dativ-n sowie Akkusativ-n-formen zur Nominativform: z.B. Eduards und Eduarden zu Eduard:
            # first step: remove genitive-s at the end of the name: eliminate last s of each name and check whether the name without s (stored as red_name) exists in the the list of all character occurences. If that reduced form exists, the reduced form is used
            corr_s_raw_all_character_occurences = []
            for name in raw_all_character_occurences:

                if re.search("'s$", name):
                    red_name = name[:-2]
                    print(red_name)
                    print(name)
                    if red_name in raw_all_character_occurences:
                        print(name)
                        print(red_name)
                        name = red_name


                #elif re.search("s$", name):
                 #   red_name = name[:-1]
                  #  print(name)
                   # print(red_name)
                    #if red_name in raw_all_character_occurences:
                     #   print(name)
                      #  print(red_name)
                       # name = red_name

                corr_s_raw_all_character_occurences.append(name)


            # second step: remove ancient dativ or accusativ "n" (for example: "Ottilien" <- "Ottilie")
            # at the end of the name: eliminate last n of each name and check whether the name without n (stored as red_name) exists in the the list of all character occurences. If that reduced form exists, the reduced form is used
            corr_raw_all_character_occurences = []
            for name in corr_s_raw_all_character_occurences:
                if re.search("n$", name):
                    red_name = name[:-1]
                    if red_name in raw_all_character_occurences:
                        name = red_name
                corr_raw_all_character_occurences.append(name)


            # ersetze Namen, die aus einem Wort bestehen, durch den vollständigen mehrteiligen Namen

            # erster Schritt: zwei neue Listen: der Ein-Wort-Namen und der Mehrwort-Namen:
            for name in corr_raw_all_character_occurences:
                name_parts = name.split(" ")
                if len(name_parts) > 1:
                    raw_composed_names.append(name)
                else:
                    raw_one_word_names.append(name)


            # zweiter Schritt: vereinheitliche Mehr-Wort-Namen hinsichtlich der Titelformen "Herrn" zu "Herr" usw.


            for name in raw_one_word_names:
                if name in " ".join(raw_composed_names):
                    for composed_name in raw_composed_names:
                        if name in composed_name:
                            paragr_characters_corr.append(composed_name)
                elif name not in " ".join(raw_composed_names):
                        paragr_characters_corr.append(name)


        counter_all_occurences = Counter(corr_raw_all_character_occurences)

        less_frq_name_to_most_frq_name_dict = {}

        if reduce_to_one_name == True:
            for character in corr_raw_all_character_occurences:
                name_parts = character.split(" ")
                if len(name_parts) > 1:
                    new_counter_dict = {}
                    for part in name_parts:
                        new_counter_dict[part] = counter_all_occurences[part]

                    more_frequent_name = max(new_counter_dict, key=new_counter_dict.get)
                    if more_frequent_name in ["Herr", "Herrn", "Don", "Graf", "Gräfin", "Frau"]:
                        false_more_freq_name = more_frequent_name
                        more_frequent_name = character.replace(false_more_freq_name, "")
                        less_frequent_name = false_more_freq_name
                    else:
                        less_frequent_name = character.replace(more_frequent_name, "")
                        less_frequent_name = less_frequent_name.replace(" ", "")
                    less_frq_name_to_most_frq_name_dict[less_frequent_name] = more_frequent_name
            print("less to most frequent name dictionary:", less_frq_name_to_most_frq_name_dict)

            # standardize names which consist of more than one word to the most frequent name part
            for paragraph in raw_nested_character_occurence:
                standard_all_character_occurences = []
                for character in paragraph:
                    name_parts = character.split(" ")
                    if len(name_parts) > 1:
                        for part in name_parts:

                            if part in less_frq_name_to_most_frq_name_dict.values():
                                standard_all_character_occurences.append(part)
                    # and replace less frequent name parts with the most frequent name
                    elif len(name_parts) == 1:
                        if character in list(less_frq_name_to_most_frq_name_dict.keys()):
                            standard_all_character_occurences.append(less_frq_name_to_most_frq_name_dict[character])
                        else:
                            standard_all_character_occurences.append(character)

                characters_in_paragraph_set = list(set(standard_all_character_occurences))
                print(characters_in_paragraph_set)
                final_characters_list += characters_in_paragraph_set
                characters_pairs_in_paragraph = list(combinations(characters_in_paragraph_set, 2))
                characters_pairs_in_paragraph = [tuple(sorted(pair)) for pair in characters_pairs_in_paragraph]
                if characters_pairs_in_paragraph:
                    character_pairs_global += characters_pairs_in_paragraph

        elif reduce_to_one_name == False:
            for paragraph in raw_nested_character_occurence:
                print(paragraph)
                characters_in_paragraph_set = list(set(paragraph))
                final_characters_list += characters_in_paragraph_set

                characters_pairs_in_paragraph = list(combinations(characters_in_paragraph_set, 2))
                characters_pairs_in_paragraph = [tuple(sorted(pair)) for pair in characters_pairs_in_paragraph]
                if characters_pairs_in_paragraph:
                    character_pairs_global += characters_pairs_in_paragraph

        print(character_pairs_global)


        self.characters_counter = Counter(final_characters_list)
        self.characters_list = [character for character, count in self.characters_counter.items() if count >= self.minimal_reference]
        self.character_pairs_counter = Counter(character_pairs_global)
        self.character_pairs_rel_freq = [(tuple[0], tuple[1], (count/len(self.chunks))) for tuple, count in self.character_pairs_counter.items() if count >= self.minimal_reference]
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(self.character_pairs_rel_freq)


    def proportion_of_characters_with_degree(self, value_degree_centrality=1):

        list_of_deg_characters = []
        degree_centrality = nx.degree_centrality(self.graph)
        for name, value in degree_centrality.items():
            if value >= value_degree_centrality :
                list_of_deg_characters.append(name)
        if len(degree_centrality) == 0:
            proportion = 0
        else:
            proportion = len(list_of_deg_characters) / len(degree_centrality)

        return proportion

    def generate_graph_centrality(self):
        if not self.characters_dict:
            print("characters_dict attribute is None. Centrality calculation expects characters_dict to be generated with generate_characters_dict method")
        else:
            centrality_dict = {}
            G = nx.Graph()
            G.add_weighted_edges_from(self.character_pairs_rel_freq)
            weighted_tuples = G.degree(weight="weight")
            sorted_weighted_tuples = sorted(weighted_tuples, key=lambda tup: tup[1], reverse=True)
            density = nx.density(G)
            degree_centrality = nx.degree_centrality(G)
            centrality_dict["weighted_degree_centrality"] = nx.degree(G, weight="weight")
            centrality_dict["deg_centr"] = degree_centrality

            self.graph_dict = centrality_dict

        pass


