'''
Created on Jan 21, 2015

@author: Minh Ngoc Le
'''
import cPickle
import os
import re
import sys
from operator import itemgetter
from nltk.corpus import wordnet as wn
from scipy.stats.stats import spearmanr
import numpy as np
from collections import defaultdict


# put path to wordnet-mlj12-definitions.txt here
def_path = '/home/minhle/scratch/wordnet-mlj12/wordnet-mlj12-definitions.txt'
assert os.path.exists(def_path)
data_path = '../data'
assert os.path.exists(data_path)


with open(os.path.join(data_path, 'WN_synset2idx.pkl')) as f:
    sys.stderr.write('Reading mapping %s... ' %data_path)
    mlj12synset2idx = cPickle.load(f)
    sys.stderr.write('Done\n')
#     print list(mlj12synset2idx.keys())[:10]

with open(def_path) as f:
    sys.stderr.write('Reading definitions %s... ' %def_path)
    synset2idx = {}
    for line in f:
        chunks = line.split('\t')
        mlj12synset, name = chunks[0], chunks[1]
        m = re.match('^__(.+)_(\w+)_(\d+)$', name)
        pos = m.group(2).replace('NN', 'n').replace('VB', 'v').replace('JJ', 'a').replace('RB', 'r')
        synset = wn.synset('%s.%s.%02d' %(m.group(1), pos, int(m.group(3))))
        assert synset
        synset2idx[synset] = mlj12synset2idx[mlj12synset]
    sys.stderr.write('Done\n')
#     print list(synset2idx.keys())[:10]
#     print list(synset2idx.values())[:10]

home_dir = os.path.dirname(__file__)
simlex_path = os.path.join(home_dir, 'SimLex-999.txt')
with open(simlex_path) as f:
    sys.stderr.write('Reading dataset from %s... ' %simlex_path)
    f.readline() # skip headders
    data = [(fields[0], fields[1], fields[2].lower(), float(fields[3])) 
            for fields in (line.strip().split('\t') for line in f)]
    sys.stderr.write('Done\n')
data_by_pos = defaultdict(list)
for point in data:
    data_by_pos[point[2]].append(point)


def sim_lemmas(sim, lemma1, lemma2, pos):
    synsets1 = wn.synsets(lemma1, pos)
    synsets2 = wn.synsets(lemma2, pos)
    if not synsets1 or not synsets2:
        return None
    return max(sim(synset2idx[s1], synset2idx[s2]) 
               if s1 in synset2idx and s2 in synset2idx
               else None
               for s1 in synsets1 for s2 in synsets2)


def _safe_spearmanr(a, b):
    indices = [i for i in range(len(a)) 
               if (a[i] is not None) and (b[i] is not None)]
    if len(indices) < len(a):
        sys.stderr.write("Omitted %d pairs\n" %(len(a)-len(indices)))
    get = itemgetter(*indices)
    a, b = get(a), get(b)
    return spearmanr(a, b)[0], len(a)


def _evalute_on_dataset(sim, data):
    _, _, _, gold = zip(*data)
    predicted = [sim_lemmas(sim, lemma1, lemma2, pos) 
                 for lemma1, lemma2, pos, _ in data]
    return _safe_spearmanr(predicted, gold)


def evaluate(path):
    with open(path) as f:
        embeddings = cPickle.load(f)
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        embeddings = embeddings.E.get_value()
#         print embeddings
    def embsim(s1, s2):
        return -np.sum((embeddings[:,s1]-embeddings[:,s2])**2)
    ret = ([['all'] + list(_evalute_on_dataset(embsim, data))] +
           [[pos] + list(_evalute_on_dataset(embsim, data_by_pos[pos])) 
            for pos in data_by_pos])
    for part, score, num in ret:
        print "%s\t%.4f\t(%d pairs)" %(part, score, num)
    

if __name__ == '__main__':
    for model in ['WN_TransE', 'WN_SE', 'WN_SME_bil', 
                  'WN_SME_lin', 'WN_Unstructured']:
        print "%s:" %model
        evaluate('%s/best_valid_model.pkl' %model)
        print