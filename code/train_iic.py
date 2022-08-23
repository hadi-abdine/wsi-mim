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
parse.add_argument('--train_path', type=str, default='./data/SemEval-2013/contexts/vectors/train_vectors_roberta17_.txt')
parse.add_argument('--test_path', type=str, default='./data/SemEval-2013/contexts/vectors/test_vectors_roberta17_.txt')
parse.add_argument('--train_mask_path', type=str, default='./data/SemEval-2013/contexts/vectors/train_vectors_roberta17_masks4.txt')
parse.add_argument('--test_mask_path', type=str, default='./data/SemEval-2013/contexts/vectors/test_vectors_roberta17_masks4.txt')
parse.add_argument('--text_path', type=str, default='./data/SemEval-2013/contexts/txt-train')
parse.add_argument('--batch_size', type=int, default=64)
parse.add_argument('--training_epochs', type=int, default=5)
parse.add_argument('--dataset', type=str, default='semeval-2013')
parse.add_argument('--eval_path', type=str, default='./data/SemEval-2013/scoring/')
parse.add_argument('--gold_path', type=str, default='./data/SemEval-2013/keys/gold/all.key')
parse.add_argument('--ids_list_path', type=str, default='./data/SemEval-2013/contexts/ids/')
parse.add_argument('--output_vectors_path', type=str, default="./data/SemEval-2013/results/iic_roberta17_7r_masks4_vectors_bestts12_tmp.key")
parse.add_argument('--output_clusters_path', type=str, default="../data/SemEval-2013/results/iic_roberta17_7r_masks4_bestts12_tmp.key")
parse.add_argument('--iic_dim', type=int, default=24)
parse.add_argument('--nb_clusters', type=int, default=7)
parse.add_argument('--dynamic', type=int, default=0)
args = parse.parse_args()
print(args.dynamic)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
training_epochs = args.training_epochs

code_path = os.path.dirname(os.path.realpath(__file__))
checkpoints_path = os.path.join(code_path, '../modelIIC')
p = args.text_path
files = [f.split('.')[0]+'.'+f.split('.')[1] for f in listdir(p) if isfile(join(p, f))]


write_path_init = os.path.dirname(args.output_vectors_path)
if not os.path.exists(write_path_init):
    os.makedirs(write_path_init)
write_path_init = os.path.dirname(os.path.join(code_path, args.output_clusters_path))
if not os.path.exists(write_path_init):
    os.makedirs(write_path_init)
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)
    


train_vectors_path_masks = args.train_mask_path
train_vectors_path = args.train_path
test_vectors_path = args.test_path
test_vectors_path_masks = args.test_mask_path

if args.dataset == 'semeval-2013':
    ids_path = args.ids_list_path
a = torch.nn.Softmax()
res_path = os.path.join(code_path, args.output_clusters_path)
vec_path = args.output_vectors_path
f = open(res_path, 'w')
f2 = open(vec_path, 'w')

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

losses = []
for ix in range(len(files)):   
    print("assign clusters for the word ",ix+1,"/"+str(len(files))+": ", files[ix])
    C = args.iic_dim

    test_vectors = getWordVectors(files[ix], path=test_vectors_path)
    test_vectorst = getWordVectors(files[ix], path=test_vectors_path_masks)
    vectors = getWordVectors(files[ix], path=train_vectors_path)
    vectorst = getWordVectors(files[ix], path=train_vectors_path_masks)
    if args.dataset == 'semeval-2013':
        fids = open(ids_path+files[ix]+".txt")   
        ids = fids.readlines()
        fids.close()
    else: ids = None

    data, X_train, X_val = prepare_data(vectors, vectorst)
    X_val, idxs = prepare_test_data(test_vectors, test_vectorst)
    dim = len(test_vectors[0])
    
    model = Model(C, dim)
    model = model.to(device)
    test_data = [(v, 0) for v in test_vectors]
    DATA = get_loader(test_data, len(test_data))

    _ = trainModel(model, data, X_train, X_val, batch_size, training_epochs, C, device,
               files[ix], checkpoints_path, printEvery=10, model_path=None, test=DATA,
                set=args.dataset, ids=ids, eval_path=args.eval_path, gold_path=args.gold_path)
    losses.append(_)
    i=0
    for batch in DATA:
        out = model.forward(batch[0])
        for v in out[0]:
            if args.dataset=='semeval-2013':
                f.write(files[ix]+' '+ids[i][:-1])
            elif args.dataset=='semeval-2010':
                f.write(files[ix]+' '+files[ix]+'.'+str(i+1))
            probas = v.tolist()
            maxi = np.argmax(probas)
            f.write(' cluster.'+str(maxi+1)+'/'+str(probas[maxi]))
            if probas[maxi] < 0.8:
                maxi2 = np.array(probas).argsort()[-2]
                f.write(' cluster.'+str(maxi2+1)+'/'+str(probas[maxi2]))
                if probas[maxi]+probas[maxi2] < 0.9:
                    maxi3 = np.array(probas).argsort()[-3]
                    f.write(' cluster.'+str(maxi3+1)+'/'+str(probas[maxi3]))
            f.write('\n')
            i += 1  
        for v in out[1]:
            f2.write(files[ix]+" - "+','.join([str(w) for w in v.tolist()])+'\n')       
    del model

f.close()
f2.close()

if args.dataset=='semeval-2013':
    f_1 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-nmi.jar") + " "+args.gold_path+" "+res_path)
    f_2 = os.popen("java -jar "+os.path.join(args.eval_path, "fuzzy-bcubed.jar") + " "+args.gold_path+" "+res_path)
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
#    print('After IIC: F-NMI = ', fn, " ; F-BC = ", fb, ' ; AVG = ', sc, '\n')
elif args.dataset=='semeval-2010':
    f = os.popen('java -jar '+os.path.join(args.eval_path, 'vmeasure.jar')+' '+res_path+' '+args.gold_path+' all')
    vm = float(f.read().split(':')[-1][:-1])
    f = os.popen('java -jar '+os.path.join(args.eval_path, 'fscore.jar')+' '+res_path+' '+args.gold_path+' all')
    fs = float(f.read().split(':')[-1][:-1])
#    print('After IIC V-measure: ', vm, 'F-score: ', fs, 'Average: ', geo_mean([vm, fs]), '\n')


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

path_vectors = vec_path
path_vectors_before = args.test_path
res_path_hard = res_path+'_hard'
res_path_soft = res_path+'_soft'

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


evaluate_semeval(res_path_hard+'_before', res_path_soft+'_before', path_vectors_before, 'before')
evaluate_semeval(res_path_hard+'_after', res_path_soft+'_after', path_vectors, 'after')

print(sum(losses)/len(files))
