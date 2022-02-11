#!/usr/bin/env python
import os
import csv
import copy
import random
from ast import literal_eval
import numpy as np
import itertools
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from sklearn.cluster import KMeans

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Global Resources
ANNOT_DATA_FP = './annotated_data/annot_data.csv'
GLOVE_TOTAL_FP = './word_embeddings/glove-wiki-300d.txt'
GLOVE_MIN_FP = './word_embeddings/glove_min.csv'
W2V_TOTAL_FP = './word_embeddings/w2v-google_news-300d.bin'
W2V_MIN_FP = './word_embeddings/w2v_min.csv'
RESULTS_HEADERS = ['verb_obj_pairs', 'composite_embeddings',
                   'cluster_IDs', 'sense_IDs', 'predicted_senses', 'correct']

# System Options
BAD_OBJECTS = ['NUM', 'NNP', 'PRP', '']
# Possible embedding sets are 'glove', 'word2vec'
EMBEDDING_SETS = ['glove', 'word2vec']
COMPOSITIONS = ['addition', 'multiplication', 'combined']
# EMBEDDING_SETS = ['glove']
# COMPOSITIONS = ['addition']
NORMALIZE_VECTORS = False


def main():
    # Load annotated data
    annot_data_dict = read_annotated_data(ANNOT_DATA_FP)
    min_vocab = set([])
    for verb in annot_data_dict:
        min_vocab.add(verb)
        for sample in annot_data_dict[verb]['sample_list']:
            min_vocab.add(sample[2])

    # Load word embeddings
    for es in EMBEDDING_SETS:
        print "********** ", es, " **********"
        vecs = load_embeddings(es, min_vocab)
        for comp in COMPOSITIONS:
            print " Beginning analysis for ...", es, " -- ", comp
            total_samples = 0
            correct_sense_assignments = 0
            # For each verb, for each object

            results_dict = {}
            for key in RESULTS_HEADERS:
                results_dict[key] = []
            results_dict['num_correct'] = 0
            for verb in annot_data_dict:
                verb_dict = {}
                verb_dict['sense_IDs'] = []
                verb_dict['verb_obj_pairs'] = []
                verb_dict['composite_embeddings'] = []
                for sample in annot_data_dict[verb]['sample_list']:
                    # Calculate composite embedding
                    sense_ID = sample[1]
                    obj = sample[2]
                    verb_we = vecs[verb]
                    obj_we = vecs[obj]
                    composite_we = compose_word_embeddings(
                        comp, verb_we, obj_we)
                    verb_dict['sense_IDs'].append(sense_ID)
                    verb_dict['verb_obj_pairs'].append((verb, obj))
                    verb_dict['composite_embeddings'].append(composite_we)

                # Perform k-means clustering
                num_senses = len(annot_data_dict[verb]['sense_ID_list'])
                kmeans = KMeans(
                    n_clusters=num_senses, precompute_distances='auto', copy_x=True, n_jobs=-1)
                cluster_IDs = kmeans.fit_predict(
                    verb_dict['composite_embeddings'])
                print "     ", verb
                for indx in range(len(verb_dict['verb_obj_pairs'])):
                    print "         ", cluster_IDs[indx], verb_dict['verb_obj_pairs'][indx]
                verb_dict['cluster_IDs'] = cluster_IDs
                verb_dict['predicted_senses'] = None
                verb_dict['num_correct'] = 0
                verb_dict['correct'] = []
                # Find best alignment of clusters to sense_IDs and record in verb dict
                alignments = list(itertools.permutations(
                    annot_data_dict[verb]['sense_ID_list']))
                for al in alignments:
                    predicted_senses = []
                    num_correct = 0
                    correct = []
                    for indx in range(len(verb_dict['verb_obj_pairs'])):
                        predicted_sense = al[cluster_IDs[indx]]
                        predicted_senses.append(predicted_sense)
                        correct_val = False
                        if predicted_sense == verb_dict['sense_IDs'][indx]:
                            num_correct += 1
                            correct_val = True
                        correct.append(correct_val)
                    if num_correct >= verb_dict['num_correct']:
                        verb_dict['num_correct'] = num_correct
                        verb_dict['correct'] = correct
                        verb_dict['predicted_senses'] = predicted_senses
                # Append verb-specific data to integrated results dictionary
                results_dict['sense_IDs'].extend(verb_dict['sense_IDs'])
                results_dict['verb_obj_pairs'].extend(
                    verb_dict['verb_obj_pairs'])
                results_dict['composite_embeddings'].extend(
                    verb_dict['composite_embeddings'])
                results_dict['cluster_IDs'].extend(verb_dict['cluster_IDs'])
                results_dict['predicted_senses'].extend(
                    verb_dict['predicted_senses'])
                results_dict['num_correct'] += verb_dict['num_correct']
                results_dict['correct'].extend(verb_dict['correct'])
            print "     Accuracy: ", float(
                results_dict['num_correct']) / float(len(results_dict['correct']))
            suffix = ".csv"
            if NORMALIZE_VECTORS:
                suffix = "_norm.csv"
            results_fp = "./results/results_" + es + "_" + comp + suffix
            write_dict_to_csv(results_dict, results_fp)


def compose_word_embeddings(composition_type, vec1, vec2):
    new_vec = None
    if composition_type == 'addition':
        new_vec = [sum(x) for x in zip(vec1, vec2)]
    elif composition_type == 'multiplication':
        new_vec = np.multiply(vec1, vec2)
    elif composition_type == 'combined':
        new_vec_mult = np.multiply(vec1, vec2)
        new_vec = [sum(x) for x in zip(vec1, vec2, new_vec_mult)]
    if NORMALIZE_VECTORS:
        new_vec = normalize_vector(new_vec)
    return new_vec


def normalize_vector(vec):
    vec_array = np.asarray(vec)
    new_vec = vec / np.sqrt((np.sum(vec_array**2)))
    return new_vec


def load_embeddings(we_set, min_vocab):
    print "     Loading word embeddings ..."
    vecs_min_fp = None
    if we_set == 'glove':
        vecs_min_fp = GLOVE_MIN_FP
    elif we_set == 'word2vec':
        vecs_min_fp = W2V_MIN_FP
    vecs = load_min_embeddings(vecs_min_fp)
    vecs_delta = []
    append_vecs = False
    for word in min_vocab:
        if word not in vecs:
            append_vecs = True
            vecs_delta.append(word)
    if append_vecs:
        print "     Appending additional word embeddings ..."
        total_vecs = load_word_embeddings_from_src(we_set)
        min_embeddings = []
        for word in vecs_delta:
            print "         --", word
            if word in total_vecs:
                we = total_vecs[word]
                if we_set == 'word2vec':
                    we = [float(x) for x in we]
                min_embeddings.append([word, we])
            else:
                print "     Error: Word not in word embedding set --", word
        write_to_csv(min_embeddings, vecs_min_fp, 'append')
        vecs = load_min_embeddings(vecs_min_fp)
    return vecs


def load_min_embeddings(fp):
    embedding_strings = read_csv(fp)
    vectors = {}
    for row in embedding_strings:
        we = literal_eval(row[1])
        vectors[row[0]] = [float(x) for x in we]
    return vectors


def read_annotated_data(fp):
    lmtzr = WordNetLemmatizer()
    annots = read_csv(fp)
    annot_data_dict = {}
    for indx, annot in enumerate(annots):
        if annot[4] not in BAD_OBJECTS:
            verb = annot[2]
            sense_ID = annot[3]
            objects = annot[4].split(',')
            if verb not in annot_data_dict:
                if verb == '':
                    print "     Blank verb is row: ", indx
                annot_data_dict[verb] = {}
                annot_data_dict[verb]['sense_ID_list'] = []
                annot_data_dict[verb]['sample_list'] = []
            for obj in objects:
                obj_lem = str(lmtzr.lemmatize(obj.strip(), 'n')).lower()
                sample_list = annot_data_dict[verb]['sample_list']
                sample = (verb, sense_ID, obj_lem)
                if sample not in sample_list:
                    sample_list.append(sample)
                    if sense_ID not in annot_data_dict[verb]['sense_ID_list']:
                        annot_data_dict[verb]['sense_ID_list'].append(sense_ID)
    return annot_data_dict


def load_word_embeddings_from_src(we_set):
    vectors = None
    if we_set == 'glove':
        fp = GLOVE_TOTAL_FP
        with open(fp, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]
    elif we_set == 'word2vec':
        fp = W2V_TOTAL_FP
        vectors = gensim.models.KeyedVectors.load_word2vec_format(
            W2V_TOTAL_FP, binary=True)
    return vectors


def read_csv(fp):
    data = []
    with open(fp, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    headers = data.pop(0)
    return data


def write_to_csv(data, fp, mode):
    write_mode = None
    if mode == 'overwrite':
        write_mode = 'wb'
    if mode == 'append':
        write_mode = 'ab'
    with open(fp, write_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerows(data)
        csvfile.close()


def write_dict_to_csv(dict, fp):
    data = []
    data.append(RESULTS_HEADERS)
    for indx in range(len(dict[RESULTS_HEADERS[0]])):
        row = []
        for key in RESULTS_HEADERS:
            row.append(dict[key][indx])
        data.append(row)
    write_to_csv(data, fp, 'overwrite')


if __name__ == '__main__':
    main()
