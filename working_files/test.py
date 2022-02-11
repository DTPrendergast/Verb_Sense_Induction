#!/usr/bin/env python
import os
import csv
import copy
import random
from ast import literal_eval
import numpy as np
import itertools
import gensim

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
RESULTS_HEADERS = ['verb_obj_pairs', 'composite_embeddings', 'cluster_IDs', 'sense_IDs', 'predicted_senses', 'correct']


def main():
    w2v_vecs = gensim.models.KeyedVectors.load_word2vec_format(W2V_TOTAL_FP, binary=True)
    # print 'besuboru'
    # print w2v_vecs['besuboru']
    print "Testing word2vec ...."
    print '     catchup'
    print '     ', w2v_vecs['catchup']

    fp = GLOVE_TOTAL_FP
    with open(fp, 'r') as f:
        glove_vecs= {}
        for line in f:
            vals = line.rstrip().split(' ')
            glove_vecs[vals[0]] = [float(x) for x in vals[1:]]
    print 'Testing glove'
    print '     catchup'
    print '     ', glove_vecs['catchup']
    print '     catch-up'
    print '     ', glove_vecs['catch-up']




if __name__ == '__main__':
    main()
