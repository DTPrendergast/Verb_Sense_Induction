#!/usr/bin/env python
import os
import csv
import copy
import random
from ast import literal_eval
import numpy as np
import nltk
from nltk.corpus import propbank
from nltk.corpus import treebank
from nltk.stem.wordnet import WordNetLemmatizer

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Global variables
ANNOT_DATA_FP = './annot_data.csv'

def main():
    lmtzr = WordNetLemmatizer()
    pb_annotations = propbank.instances()
    annot_data = []
    # for indx in range(0, 100):
    #     annot = pb_annotations[indx]
    for indx, annot in enumerate(pb_annotations):
        tree = annot.tree
        args = annot.arguments
        if (indx % 100)==0:
            print indx
        # assert tree == treebank.parsed_sents(annot.fileid)[annot.sentnum]
        if tree is None:
            continue
        for (argloc, argid) in args:
            if argid=="ARG1":
                sent = ' '.join([w for w in tree.leaves()])
                pred_word = annot.predicate.select(tree).leaves()[0]
                pred_lem = lmtzr.lemmatize(pred_word, 'v')
                sense_ID = annot.roleset
                object = argloc.select(tree)
                object_str = ' '.join(str(object).split())
                annot_data.append([sent, pred_word, pred_lem, sense_ID, object_str])

    print "done finding preds and info.  Writing ..."
    write_results(annot_data, ANNOT_DATA_FP)

def write_results(data, fp):
    with open(fp, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerows(data)

if __name__ == '__main__':
    main()
