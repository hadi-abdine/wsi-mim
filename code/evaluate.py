from os import listdir
from os.path import isfile, join
from senses_functions import *
from IICModel import *
import warnings
warnings.filterwarnings('ignore')
import argparse
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os

parse = argparse.ArgumentParser()
parse.add_argument('--vectors_path', type=str, default='../data/SemEval-2013/contexts/vectors/train_vectors_roberta17_.txt')
parse.add_argument('--dataset', type=str, default='semeval-2013')
parse.add_argument('--eval_path', type=str, default='../data/SemEval-2013/scoring/')
parse.add_argument('--gold_path', type=str, default='../data/SemEval-2013/keys/gold/all.key')
parse.add_argument('--ids_list_path', type=str, default='./data/SemEval-2013/contexts/ids/')
parse.add_argument('--nb_clusters', type=int, default=7)
parse.add_argument('--dynamic', type=int, default=0)
parse.add_argument('--files_path', type=str, default='../data/SemEval-2013/contexts/txt-test/')
args = parse.parse_args()

p = args.files_path
files = [f.split('.')[0]+'.'+f.split('.')[1] for f in listdir(p) if isfile(join(p, f))]
code_path = os.path.dirname(os.path.realpath(__file__))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

path_vectors = args.vectors_path
res_path_hard = 'hard_clustering.key'
res_path_soft = 'soft_clustering.key'
ids_path = args.ids_list_path

def evaluate_semeval(res_path_hard, res_path_soft, path_vectors, state):
    f_c = open(res_path_hard, 'w')
    f_c_s = open(res_path_soft, 'w')

    if args.dataset=='semeval-2013': scores_path = open(os.path.join(code_path, '../data/pyramid_scores/test_2013/pca3_L8'),'r')
    if args.dataset=='semeval-2010': scores_path = open(os.path.join(code_path, '../data/pyramid_scores/test_2010/pca3_L8'),'r')
    scores = dict()
    for line in scores_path.readlines():
        line = line[:-1]
        scores[line.split(',')[0]] = float(line.split(',')[1])
    scores_path.close()
    maxi_c = max(scores.values())
    mini_c = min(scores.values())

    all_probs = []
    for ix in range(len(files)):
        test_vectors = preprocessing.normalize(getWordVectors(files[ix], path=path_vectors))
        if args.dynamic==1:
            k = int(4*(maxi_c-scores[files[ix]])/(maxi_c-mini_c) + 4)
        else:
            k = args.nb_clusters
        clustering = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average').fit(test_vectors)
        clusters = dict()
        for i, com in enumerate(clustering.labels_):
            clusters[i] = com
        centers = get_cluster_vector(files[ix], clusters, path=path_vectors)
        v_probs = []
        dists = []
        t = 0.03
        for vect in test_vectors:
            probs = []
            dists_tmp = []
            for i in range(len(set(clustering.labels_))):
                dist = np.linalg.norm(np.array(vect)-np.array(centers[i]))
                dists_tmp.append(dist)
                if dist == 0: probs.append(1000000)
                else: probs.append(1/dist)
            v_probs.append(softmax([el / t for el in normalize(probs)]))
            dists.append(dists_tmp)
            all_probs += list(v_probs[-1])
        if args.dataset=='semeval-2013':
            fids = open(ids_path+files[ix]+".txt")
            ids = fids.readlines()
            fids.close()
        
        for i, c in enumerate(clustering.labels_):
            if i<len(test_vectors):
                if args.dataset=='semeval-2013':
                    f_c.write(files[ix]+' '+ids[i][:-1]+' '+'cluster.'+str(c+1)+'\n')
                elif args.dataset=='semeval-2010':
                    f_c.write(files[ix]+' '+files[ix]+'.'+str(i+1)+' '+'cluster.'+str(c+1)+'\n')

        all_probs.sort(reverse=True)
        for i, v in enumerate(v_probs):
            if args.dataset=='semeval-2013':
                f_c_s.write(files[ix]+' '+ids[i][:-1])
            elif args.dataset=='semeval-2010':
                f_c_s.write(files[ix]+' '+files[ix]+'.'+str(i+1))
            probas = v
            maxi = np.argmax(probas)
            f_c_s.write(' cluster.'+str(maxi+1)+'/'+str(probas[maxi]))
            if probas[maxi] < 0.8:
                maxi2 = np.array(probas).argsort()[-2]
                f_c_s.write(' cluster.'+str(maxi2+1)+'/'+str(probas[maxi2]))
                if probas[maxi]+probas[maxi2] < 0.85:
                    maxi3 = np.array(probas).argsort()[-3]
                    f_c_s.write(' cluster.'+str(maxi3+1)+'/'+str(probas[maxi3]))
            f_c_s.write('\n')

    f_c.close()
    f_c_s.close()

    if args.dataset=='semeval-2013':
        f_1 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-nmi.jar") + " "+args.gold_path+" "+res_path_hard)
        f_2 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-bcubed.jar") + " "+args.gold_path+" "+res_path_hard)
        for line in f_1.read().split('\n'):
            line=line.replace(' ', '\t')
            line=line.replace('\t\t', '\t')
            line=line.replace('\t\t', '\t')
            if line.split('\t')[0]=='all':
                fn = float(line.split('\t')[1])
                break
        for line in f_2.read().split('\n'):
            line=line.replace(' ', '\t')
            line=line.replace('\t\t', '\t')
            line=line.replace('\t\t', '\t')
            if line.split('\t')[0]=='all':
                fb = float(line.split('\t')[3])
                break
        sc = geo_mean([fn, fb])
        print('Hard Agglomerative Clustering'+state+' MIM: F-NMI = ', fn, " ; F-BC = ", fb, ' ; AVG = ', sc, '\n')

        f_1 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-nmi.jar") + " "+args.gold_path+" "+res_path_soft)
        f_2 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-bcubed.jar") + " "+args.gold_path+" "+res_path_soft)
        for line in f_1.read().split('\n'):
            line=line.replace(' ', '\t')
            line=line.replace('\t\t', '\t')
            line=line.replace('\t\t', '\t')
            if line.split('\t')[0]=='all':
                fn = float(line.split('\t')[1])
                break
        for line in f_2.read().split('\n'):
            line=line.replace(' ', '\t')
            line=line.replace('\t\t', '\t')
            line=line.replace('\t\t', '\t')
            if line.split('\t')[0]=='all':
                fb = float(line.split('\t')[3])
                break
        sc = geo_mean([fn, fb])
        print('Soft Agglomerative Clustering '+state+' MIM: F-NMI = ', fn, " ; F-BC = ", fb, ' ; AVG = ', sc, '\n')
    elif args.dataset=='semeval-2010':
        f = os.popen('java -jar '+os.path.join(args.eval_path, 'vmeasure.jar')+' '+res_path_hard+' '+args.gold_path+' all')
        vm = float(f.read().split(':')[-1][:-1])
        f = os.popen('java -jar '+os.path.join(args.eval_path, 'fscore.jar')+' '+res_path_hard+' '+args.gold_path+' all')
        fs = float(f.read().split(':')[-1][:-1])
        print('Hard Agglomerative Clustering '+state+' MIN: V-measure: ', vm, 'F-score: ', fs, 'Average: ', geo_mean([vm, fs]), '\n')
        f = os.popen('java -jar '+os.path.join(args.eval_path, 'vmeasure.jar')+' '+res_path_soft+' '+args.gold_path+' all')
        vm = float(f.read().split(':')[-1][:-1])
        f = os.popen('java -jar '+os.path.join(args.eval_path, 'fscore.jar')+' '+res_path_soft+' '+args.gold_path+' all')
        fs = float(f.read().split(':')[-1][:-1])
#        print('Soft Agglomerative Clustering '+state+' MIM: V-measure: ', vm, 'F-score: ', fs, 'Average: ', geo_mean([vm, fs]), '\n')


evaluate_semeval(res_path_hard+'_after', res_path_soft+'_after', path_vectors, 'after')
