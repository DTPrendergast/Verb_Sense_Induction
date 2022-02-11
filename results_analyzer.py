#!/usr/bin/env python
import os
import csv
import copy
import random
from ast import literal_eval
import numpy as np
import itertools
import random


# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Global Resources
RESULTS_HEADERS = ['verb_obj_pairs', 'composite_embeddings', 'cluster_IDs', 'sense_IDs', 'predicted_senses', 'correct']

# System Options
NUM_ITERS = 10000
EMBEDDING_SETS = ['glove', 'word2vec']
COMPOSITIONS = ['addition', 'multiplication', 'combined']
NORMALIZE_VECTORS = False


def main():
    for es in EMBEDDING_SETS:
        for comp in COMPOSITIONS:
            suffix = ".csv"
            if NORMALIZE_VECTORS:
                suffix = "_norm.csv"
            results_fp = "./results/results_" + es + "_" + comp + suffix
            results = read_csv_as_dict(results_fp, None)
            for indx, pair in enumerate(results['verb_obj_pairs']):
                results['verb_obj_pairs'][indx] = literal_eval(pair)
            num_correct = results['correct'].count('True')
            total_samples = len(results['correct'])
            true_accuracy = float(num_correct)/float(total_samples)

            verb_sense_dict = get_verb_senses(results)
            # for verb in verb_sense_dict:
            #     print verb, verb_sense_dict[verb]
            random_total = 0
            random_correct = 0
            for _i in range(NUM_ITERS):
                correct_sense_IDs = results['sense_IDs']
                for indx, row in enumerate(results['verb_obj_pairs']):
                    random_total += 1
                    verb = row[0]
                    sense_ID_list = verb_sense_dict[verb]
                    random_sense_ID = random.choice(sense_ID_list)
                    if random_sense_ID==correct_sense_IDs[indx]:
                        random_correct += 1
            random_accuracy = float(random_correct)/float(random_total)
            # print random_total
            # print random_correct
            print es, "--", comp
            print "     True Accuracy: ", true_accuracy
            print "     Random Accuracy: ", random_accuracy

def get_verb_senses(results):
    dict = {}
    for indx, row in enumerate(results['verb_obj_pairs']):
        verb = row[0]
        if verb not in dict:
            dict[verb] = []
        sense_ID = results['sense_IDs'][indx]
        if sense_ID not in dict[verb]:
            dict[verb].append(sense_ID)
    # for verb in dict:
    #     print verb
    #     print "     ", dict[verb]
    return dict

def array_2_dict(data, headers):
    dict = {}
    if headers is None:
        headers = data.pop(0)
    for hdr in headers:
        indx = headers.index(hdr)
        dict[hdr] = []
        for row in data:
            dict[hdr].append(row[indx])
    return dict

def read_csv(fp):
    data = []
    with open(fp,'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    return data

def read_csv_as_dict(fp, headers):
    data_as_lists = read_csv(fp)
    data_as_dict = array_2_dict(data_as_lists, headers)
    return data_as_dict

def write_to_csv(data, fp, mode):
    write_mode = None
    if mode=='overwrite':
        write_mode = 'wb'
    if mode=='append':
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
