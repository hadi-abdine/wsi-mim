import fairseq
import torch
import nltk
import argparse
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from numpy import random
from nltk.stem import WordNetLemmatizer
import os


parse = argparse.ArgumentParser()
parse.add_argument('--read_path', type=str, default='../data/SemEval-2013/contexts/txt-test/')
parse.add_argument('--write_path', type=str, default='../data/SemEval-2013/contexts/txt-test-masks4/')
parse.add_argument('--percentage', type=float, default=0.4)
args = parse.parse_args()
wnl = WordNetLemmatizer()

if not os.path.exists(args.write_path):
    os.makedirs(args.write_path)

def replace_with_lemma(lemma, pos, sentence):
    s = ''
    ind = -1
    for i, token in enumerate(sentence.replace('-',' ').replace('=', ' =').replace('.', ' ').replace("'","' ").lower().split()):
        if lemma==wnl.lemmatize('color', pos) and token=='colour':
            s += token + ' '
            ind = i
        elif lemma==wnl.lemmatize('lie', pos) and (wnl.lemmatize(token, pos)=='lay' or wnl.lemmatize(token, pos)=='lain' or token=='lah'):
            s += token + ' '
            ind = i
        elif lemma==wnl.lemmatize('figure', pos) and (token=='figger' or token=='figgered'):
            s += token + ' '
            ind = i
        elif wnl.lemmatize(token, pos) == lemma:
            s += token + ' '
            ind = i
        else:
            s += token + ' '
    return s, ind

def get_paraphrases(sentences, word):
    paraphrases = []
    pos = word[1]
    
    if pos=='j':
        pos = 'a'
    lemma = wnl.lemmatize(word[0], pos)
    print('start masking...')
    for i, sentence in enumerate(sentences[:3500]):
        sentence, ind = replace_with_lemma(lemma, pos, sentence)
        tmp = sentence.split()
        masked = False
        if len(tmp)>2:
            nb_masks = max(int(args.percentage*len(tmp)), 2)
        else:
            nb_masks = 1
        inds = random.choice(len(tmp), nb_masks, False)
        
        while not masked:
            inds = random.choice(len(tmp), nb_masks, False)
            if ind not in inds:
                masked = True
            elif len(sentence.split())==1:
                masked = True
        for j in inds:
            # if pos == 'txt':
            #     if tmp[j] != word[0] or len(set(tmp))==1:
            #         tmp[j] = '<mask>'
            #         masked = True
            # else:
                # if wnl.lemmatize(tmp[j], pos) != lemma:
            tmp[j] = '<mask>'
                    # masked = True
        sentences[i] = ' '.join(tmp) 

    print('finished masking, start predicting...')
    tmp_para = bart.fill_mask(sentences[:3500], topk=1, beam=1)
    paraphrases = [tmp_para[j][0][0] for j in range(len(sentences[:3500]))]
    print(len(paraphrases), len(sentences[:3500]))
    return paraphrases

def write_to_file(paraphrases, file_path):
    with open(file_path, 'w') as f:
        for s in paraphrases:
            f.write("%s\n" % s)


random.seed(1221996)
print(args.read_path)
bart = torch.hub.load('pytorch/fairseq', 'bart.base')
bart.eval()

p = args.read_path
p2 = args.write_path

files1 = [f for f in listdir(p) if isfile(join(p, f))]
files2 = [f for f in listdir(p2) if isfile(join(p2, f))]
files = [f for f in files1 if f not in files2]
print('files: ', files)
j = 1
for f in files:
    print("word ", j, '/' + str(len(files)) + ': ', f)
    j += 1
    with open(join(p, f), 'r') as f1:
        para = get_paraphrases(f1.readlines(), f.split('.'))
        write_to_file(para, join(args.write_path, f))
