from scipy import spatial
from sklearn.neighbors import kneighbors_graph
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.label_propagation import label_propagation_communities
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch
import os
import random
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
import pylab as plt
from sklearn.metrics.pairwise import euclidean_distances as ed
import json
import pickle

code_path = os.path.dirname(os.path.realpath(__file__))
r = 17267812 # fix the random seed for reproducibility
with open(os.path.join(code_path, '../data/stopwords-fr.json')) as f:
    fr_stop_words = json.load(f)
to_add=['dun', 'dune', 'cest', "c’est", 'quil']
for w in to_add:
    fr_stop_words.append(w)

def clusters_save(word, clusters, nature):
    with open(word + '_' + nature + '.pickle', 'wb') as f:
        pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)

def clusters_load(word, nature):
    with open(word + '_' + nature + '.pickle', 'rb') as f:
        return pickle.load(f)    
    
# Get the 3000 vectors of a word
def getWordVectors(word, remove_activation=False, path='/datasets/polysemy/all_vectors.txt'): 
    """
    Gets a list of vectors of a specific word from a file

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"

    Returns
    -------
    list
        a list of lists (vectors of a word)
    """
    f = open(path, "r")
    vectors_list = []
    while 1:
        line = f.readline()
        if(line.split()[0]==word):
            # print('found it!')
            if line.split()[2] == 'skip' or line.split()[2] == 'good' or line.split()[2] == 's,k,i,p':
                # vectors_list.append(line.split()[2])
                continue
            else:
                vectors_list.append([float(num) for num in line.split()[2].split(',')])
            
            if remove_activation:
                vectors_list[-1][np.argmax(np.abs(np.array(vectors_list[-1])))] = 0
            
            while True:
#             for j in range(3499):
                line = f.readline()
                if(line!='' and line.split()[0]==word):
                    if line.split()[2] == 'skip' or line.split()[2] == 'good' or line.split()[2] == 's,k,i,p':
                        # vectors_list.append(line.split()[2])
                        continue
                    else:
                        vectors_list.append([float(num) for num in line.split()[2].split(',')])
                    if remove_activation:
                        vectors_list[-1][np.argmax(np.abs(np.array(vectors_list[-1])))] = 0
                else:
                    break
#             vectors_list = preprocessing.normalize(vectors_list, axis=1)
            break
            
    f.close()
    return vectors_list

# Get the 3000 sentences of a word
def get_sentences(word, path='/data/home/cxypolop/Projects/openpaas_elmo/polysemy_roberta/en_extracted_sentences/'):
    """
    Gets a list of str of a specific word from a file

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    path : str
        The location that contains text files wher each one contains sentences of the word

    Returns
    -------
    list
        a list of str
    """
    with open(path+word+'.txt') as f:
        sentences = f.readlines()
    return sentences


# Get the cosine similarities between all the vectors
def getCosSim(word, remove_activation=False, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Gets a dict that contains the cosine similarity between all the vectors of a word

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"

    Returns
    -------
    list
        a dictionary where the keys are tuples of two nodes and the value is the cosine similarity
    """
    if vectors is None:
        vectors_list = getWordVectors(word, remove_activation, path)
    else:
        vectors_list = vectors.copy()
    cos_sim = dict()
    for ind1, vect1 in enumerate(vectors_list):
        for ind2, vect2 in enumerate(vectors_list[ind1+1:]):
            sec = ind1+ind2+1
            vec1 = vectors_list[ind1]
            vec2 = vectors_list[sec]
            cos_sim[(ind1, sec)] = 1 - spatial.distance.cosine(vec1, vec2) 
    return cos_sim


# Buils a graph based on cosine similarity and apply louvain community detection
def community_detection_cossim(word, th, label_propagation=False, remove_activation=False, path='/datasets/polysemy/all_vectors.txt'):
    """
    Gets the clusters of a word's senses based on building a graph of cosine similarities with threshold.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    th : int
        The minimum cosine similarity value to make an edge between two nodes
    label_propagation: bool,
        A boolean used to specify the community detection algorithm,
        --if set to True, it uses label propagation algorithm from community in network x (none weighted graph),
          ref: Community Detection via Semi-Synchtomous Label Propagation Algorithms
        --if set to False, it uses community louvain function (weighted graph),
          ref: Fast Unfolding of communities in large networks
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"

    Returns
    -------
    partition: dict
        a dictionary that assign each vector to a cluster.
    G: networkx weighted graph.
        The built cosine similarities graph.
    """
    cos_sim = getCosSim(word, remove_activation, path)
    maxi = max(cos_sim.values())
    mini = min(cos_sim.values())
    G_norm =nx.Graph()
    cos_sim_norm = dict()
    maxindex = 0
    
    for i in cos_sim.keys():
        val = 100 * ((cos_sim[i]-mini)/(maxi-mini))
        maxindex = max(maxindex, i[1])
        if val>th:
            cos_sim_norm[i] = val
    for j in range(maxindex+1):
        G_norm.add_node(j)
    G_norm.add_weighted_edges_from([(n1, n2, cos_sim_norm[(n1,n2)]) for (n1, n2) in list(cos_sim_norm.keys())])
    
    if label_propagation:
        clusters = label_propagation_communities(G_norm)
        for cluster in clusters:
            partitions.append(cluster)
        partition = dict()
        for idx, cluster in enumerate(partitions):
            for node in cluster:
                partition[node] = idx
    else:
        partition = community_louvain.best_partition(G_norm, random_state=r)
            
    return partition, G_norm

def community_detection_cossim_knn(word, k, label_propagation=False, remove_activation=False, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Gets the clusters of a word's senses based on building a graph of cosine similarities using KNN(metric=cosine similarity).

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    k : int
        The number of neighbors to consider for each node
    label_propagation: bool,
        A boolean used to specify the community detection algorithm,
        --if set to True, it uses label propagation algorithm from community in network x (none weighted graph),
          ref: Community Detection via Semi-Synchtomous Label Propagation Algorithms
        --if set to False, it uses community louvain function (weighted graph),
          ref: Fast Unfolding of communities in large networks
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset

    Returns
    -------
    partition: dict
        a dictionary that assign each vector to a cluster.
    G: networkx weighted graph.
        The built cosine similarities graph.
    """
    vectors_list =[]
    if vectors is None:
        vectors_list = getWordVectors(word, path=path)
    else:
        vectors_list = vectors.copy()
#     A = kneighbors_graph(vectors_list, k, mode='distance', p=2, include_self=False, n_jobs=-1)
#     A = A.toarray()
    
    n = len(vectors_list)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    sims = getCosSim(word, remove_activation, path, vectors=vectors_list.copy())
    maxi = max(sims.values())
    mini = min(sims.values())
    for i in sims.keys():
        B[i[0], i[1]] = 100 * ((sims[i]-mini)/(maxi-mini))
        B[i[1], i[0]] = B[i[0], i[1]]
    for i in range(n):
        indexes = B[i].argsort()[-k:][::-1]
        for j in indexes:
            A[i][j] = B[i][j]
    
#     for i in range(len(A)):
#         for j in A[i].nonzero()[0]:
#             A[i, j] = 1 - spatial.distance.cosine(vectors_list[i], vectors_list[j]) 

    G = nx.from_numpy_matrix(A)

    
    if not label_propagation:
        partition = community_louvain.best_partition(G, random_state=r)
    else:
        partitions = []
        clusters = label_propagation_communities(G)
        for cluster in clusters:
            partitions.append(cluster)
        partition = dict()
        for idx, cluster in enumerate(partitions):
            for node in cluster:
                partition[node] = idx

    return partition, G

def remove_clusters_overlap(word, cluster, th, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Combine clusters that have a high cosine similarity

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    cluster: dict
        a dictionary that assign each vector to a cluster.
    th : int
        The minimum cosine similarity value combine two nodes
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset
        

    Returns
    -------
    clusters: dict
        a dictionary that assign each vector to a cluster.
    """
    clusters = cluster.copy()
    cv = get_cluster_vector(word, clusters, path, vectors=vectors)
    vectors_list = [v for k,v in cv.items()]
    cos_sim = dict()
    for ind1, vect1 in enumerate(vectors_list):
        for ind2, vect2 in enumerate(vectors_list[ind1+1:]):
            sec = ind1+ind2+1
            vec1 = vectors_list[ind1]
            vec2 = vectors_list[sec]
            cos_sim[(ind1, sec)] = 1 - spatial.distance.cosine(vec1, vec2) 


    # for k in sorted(cos_sim, key=cos_sim.get, reverse=True):
    #     print(k, cos_sim[k])


    remove_overlap = dict()
    c_index = 0
    final_clusters = dict()
    for k, v in cos_sim.items():
        if v > th:
            found = False
            for k1, v1 in remove_overlap.items():
                if k[0] in v1 or k[1] in v1:
                    v1.append(k[0])
                    v1.append(k[1])
                    found = True
                    break
            if not found:
                remove_overlap[c_index] = [k[0], k[1]]
                c_index += 1
    n = len(remove_overlap.keys())
    for i in range(n-1):
        if i in remove_overlap.keys():
            for j in range(i+1, n):
                if j in remove_overlap.keys():
                    if len(set.intersection(set(remove_overlap[i]), set(remove_overlap[j])))>0:
                        remove_overlap[i] = set.union(set(remove_overlap[i]), set(remove_overlap[j]))
                        remove_overlap.pop(j)
    i = 0
    for k, v in remove_overlap.items():
        final_clusters[i] = set(v)
        i += 1
    for i in range(len(cv.keys())):
        found = False
        for k, v in remove_overlap.items():
            if i in v:
                found = True
                break
        if not found:
            final_clusters[len(final_clusters.keys())] = set([i])

    for k, v in clusters.items():
        for k1, v1 in final_clusters.items():
            if v in v1:
                clusters[k] = k1
                break
    
    return clusters

# draw the graph
def plot_graph(G, partition):
    """
    plot the graph with the communities

    Parameters
    ----------
    G : NetworkX graph
        The weighted graph of the vectors
    partition: dict
        a dictionary that assign each vector to a cluster.

    Returns
    -------
    None
    """
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    
    
# Build a KNN graph and apply louvain community detection
def get_communities_knn_graph(word, k=20, p1=2, label_propagation=False, remove_activation=False, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Gets the clusters of a word's senses based on building a KNN graph of distances with community detection.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    k : int
        The number of neighbors to consider for each node
    p1 : int
        The power of distance to use in building the KNN graph
    label_propagation: bool,
        A boolean used to specify the community detection algorithm,
        --if set to True, it uses label propagation algorithm from community in network x (none weighted graph),
          ref: Community Detection via Semi-Synchtomous Label Propagation Algorithms
        --if set to False, it uses community louvain function (weighted graph),
          ref: Fast Unfolding of communities in large networks
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset

    Returns
    -------
    partition: dict
        a dictionary that assign each vector to a cluster.
    G: networkx weighted graph.
        The built cosine similarities graph.
    """
    partitions = []
    if vectors is None:
        vectors_list = getWordVectors(word, remove_activation, path=path)
    else:
        vectors_list = vectors
    A = kneighbors_graph(vectors_list, k, mode='distance', p=p1, include_self=False, n_jobs=-1)
    G=nx.from_numpy_matrix(A.toarray())
    if not label_propagation:
        partition = community_louvain.best_partition(G, random_state=r)
    else:
        clusters = label_propagation_communities(G)
        for cluster in clusters:
            partitions.append(cluster)
        partition = dict()
        for idx, cluster in enumerate(partitions):
            for node in cluster:
                partition[node] = idx
    return partition, G



# Apply 2 consucetive community detection algorithms to reduce overlap between clusters
def get_communities_knn_graph_no_overlap(word, k=20, p=2, label_prop=False, cossim=False, remove_activation=False, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Gets the clusters of a word's senses performing two consicutive community detection algorithm.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors
    k : int
        The number of neighbors to consider for each node
    p : int
        The power of distance to use in building the KNN graph
    label_prop: bool,
        A boolean used to specify the community detection algorithm,
        --if set to True, it uses label propagation algorithm from community in network x (none weighted graph),
          ref: Community Detection via Semi-Synchtomous Label Propagation Algorithms
        --if set to False, it uses community louvain function (weighted graph),
          ref: Fast Unfolding of communities in large networks
    cossim: bool,
        A boolean that used to determine whether use KNN with cosine similarities or KNN with distances
    remove_Activation : bool, optional
        A boolean used to remove the peak from RoBERTa vector if it's True
    path : str
        The location that contains the vectors of the words
        format of the text: "word - float,float,float,<...> \n"
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset

    Returns
    -------
    partition: dict
        a dictionary that assign each vector to a clusterusing community louvain as the second clustering.
    partition_labelprop: dict
        a dictionary that assign each vector to a cluster using label propagation as the second clustering.
    G: networkx weighted graph.
        The built cosine similarities graph.
    """
    
    # get the communities given the method (label propagation or community louvain)
    if not label_prop:
        if not cossim:
            partition, G = get_communities_knn_graph(word, k, p, label_prop, remove_activation, path, vectors)
        else:
            partition, G = community_detection_cossim_knn(word, k, label_prop, remove_activation, path, vectors)
    else:
        if not cossim:
            partition, G = get_communities_knn_graph_label_prop(word, k, p, label_prop, remove_activation, path, vectors)
        else:
            partition, G = community_detection_cossim_knn(word, k, label_prop, remove_activation, path, vectors)
            
    partition_labelprop = partition.copy()
    # get_cluster_vector
    # build the second graph in order to perform community detection for the second time
    inv_partition = {v:k for k,v in partition.items()}
    coms = dict()
    for k, v in inv_partition.items():
        coms[k] = [node for node, val in partition.items() if val == k]
    w = dict()
    
    for i in range(len(coms.keys())-1):
        for j in range(i+1, len(coms.keys())):
            w[(i, j)] = 0
            counts = 0
            for node1 in coms[i]:
                for node2 in coms[j]:
                    if G.has_edge(node1, node2):
                        counts += 1
                        w[(i, j)] += G[node1][node2]["weight"]
                    
    
        
    G_red =nx.Graph()
    G_red.add_weighted_edges_from([(n1, n2, w[(n1,n2)]) for (n1, n2) in list(w.keys())])
    
    # doing the second community detection using community louvain
    partition2 = community_louvain.best_partition(G_red, random_state=r)
    for key in list(partition.keys()):
        partition[key] = partition2[partition[key]]
    
    # doing the second community detection using label propagation
    partitions_labelprop = [] 
    partitions = label_propagation_communities(G_red)
    for cluster in partitions:
        partitions_labelprop.append(cluster)
    partition_labelprop2 = dict()
    for idx, cluster in enumerate(partitions_labelprop):
        for node in cluster:
            partition_labelprop2[node] = idx  
    for key in list(partition_labelprop.keys()):
        aux = partition_labelprop[key]
        partition_labelprop[key] = partition_labelprop2[aux]
        
    return partition, partition_labelprop, G

# Get the list of vectors for each cluster
def get_cluster_vector(word, cluster, path='/datasets/polysemy/all_vectors.txt', vectors=None):
    """
    Gets the average vector of each cluster.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors.
    cluster : dict
        A dictionary that maps each node to a cluster.
    vectors : list
        list of vectors of a word in case you don't have a text file of vectors

    Returns
    -------
    inv_cluster: dict
        a dictionary that maps each cluster to a vector.
    """
    inv_cluster = {v:k for k,v in cluster.items()}
    coms = []
    for k, v in inv_cluster.items():
        coms.append([node for node, val in cluster.items() if val == k])
    if vectors is None:
        vects = preprocessing.normalize(getWordVectors(word, path=path))
    else:
        vects = vectors
    for i, com in enumerate(coms):
        v_temp = []
        for j in com:
            v_temp.append(vects[j])
        inv_cluster[i] = np.array(v_temp).transpose().mean(axis=1)
    return inv_cluster


# Get RoBERTa vectors of a test word
def get_test_word_vect(word, sentence):
    """
    Gets the RoBERTa vector of a word given a sentence.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors.
    sentence : str
        A string that contains the word to get its vector

    Returns
    -------
    vector: numpy array
        RoBERTa vector of word
    """
    word = word.lower()
    sentence = sentence = ' '.join(nltk.word_tokenize(sentence)).lower()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True)
#     roberta.to(device)
    roberta.eval()
    
    encoded_input = tokenizer.encode(sentence, return_tensors='pt')
    hidden_states = roberta(encoded_input)[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1).cpu()
    word_instances = []
    tokens = [token_str for token_str in tokenizer.convert_ids_to_tokens(encoded_input[0])]
    i = 1
    while i < len(tokens):
        if tokens[i]=='</s>':
            break
        if "Ġ" + word == tokens[i]:
            if tokens[i+1][0] == "Ġ" or tokens[i+1]=='</s>':
                word_instances.append([i,])
        else: 
            tmp_word = tokens[i]
            tmp_i = [i]
            while tokens[i+1][0] != "Ġ" and tokens[i+1]!='</s>':
                i += 1
                tmp_word += tokens[i]
                tmp_i.append(i)
            if tmp_word == "Ġ" + word:
                word_instances.append(tmp_i)
        i += 1    
    if len(word_instances) == 0:
        return False
    print(word_instances)
    index = random.choice(word_instances)
    vectors = [token_embeddings[-1][i].detach().numpy() for i in index]
    vector = np.mean(vectors, axis=0)
    
    return vector


# Assign a word to a cluster based on Euclidian distance
def find_test_word_cluster(word, cluster, path='/datasets/polysemy/all_vectors.txt', sentence='', vector=None, vectors=None):
    """
    Assign a word to a cluster given its vector or the word with the sentence using euclidian distance.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors.
    cluster: dict
        A dictionary that maps each node to a cluster.
    path : str
        The location of the list of vectors.
        format of the text: "word - float,float,float,<...> \n"
    sentence : str
        A string that contains the word to get its vector.
    vector : list
        The vector of the word must be given if there's no sentence
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset

    Returns
    -------
    assigned_cluster : int
        The cluster of the test word
    """
    clusters = get_cluster_vector(word, cluster, vectors=vectors)
    if vector is None:
        vector = get_test_word_vect(word, sentence)
    min_dist = 99999999
    assigned_cluster = -1
    for c, v in clusters.items():
        dist = np.linalg.norm(np.array(vector)-np.array(v))
        if min_dist > dist:
            min_dist = dist
            assigned_cluster = c
    return assigned_cluster

def find_test_word_cluster_cossim(word, cluster, sentence='', path='', vector=None, vectors=None):
    """
    Assign a word to a cluster given its vector or the word with the sentence using cosine similarity.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors.
    cluster: dict
        A dictionary that maps each node to a cluster.
    sentence : str
        A string that contains the word to get its vector.
    vector : list
        The vector of the word must be given if there's no sentence
    vectors: list, optional
        A list of RoBERTa vectors can be given instead of the word if it's not in the dataset

    Returns
    -------
    assigned_cluster : int
        The cluster of the test word
    """
    if vector is None:
        vector = get_test_word_vect(word, sentence)
    clusters = get_cluster_vector(word, cluster,path=path, vectors=vectors)
    max_cos = 0
    assigned_cluster = -1
    for c, v in clusters.items():
        cos = 1 - spatial.distance.cosine(vector, v) 
        if max_cos < cos:
            max_cos = cos
            assigned_cluster = c
    return assigned_cluster

# Get relevant words of a cluster
def get_relevant_words(word, clusters, sentences_path='/data/home/cxypolop/Projects/openpaas_elmo/polysemy_roberta/en_extracted_sentences/', portion=None):
    """
    Gets the list of unigrams sorted by frequency for each cluster.

    Parameters
    ----------
    word : str
        The word the we want to load its vectors.
    clusters : dict
        A dictionnary contains the cluster of each node.
    sentences_path : str
        The path of the folder contains <word>.txt that contains all the sentences for the word.
    portion : list, optional
        contains the indexes of the sentences that we want to take in consideration.

    Returns
    -------
    inv_cluster: dict
        A dictionnaru maps each cluster to a list of unigrams.
    """
    sentences = get_sentences(word, sentences_path)
    if portion is not None:
        sentences = [sentences[i] for i in portion]
    inv_cluster = {v:k for k,v in clusters.items()}
    coms = dict()
    for k, v in inv_cluster.items():
        coms[k] = [node for node, val in clusters.items() if val == k]
    for i, com in coms.items():
        unigrams = dict()
        for j in com:
            if j<len(sentences):
                for word in sentences[j].split():
                    word = word.lower()
                    if word not in stopwords.words('english') and word not in fr_stop_words and len(word)>2 and word != 'also':
                        if word not in unigrams:
                            unigrams[word] = 1
                        else:
                            unigrams[word] += 1
        sorted_words = [k for k in sorted(unigrams.items(), key=lambda item: item[1])]
        inv_cluster[i] = sorted_words
    return inv_cluster

def remove_peak(word_vectors):
    """
    Remove the peak from RoBERTa vectors

    Parameters
    ----------
    word_vectors : list
        List of vectors of a word.

    Returns
    -------
    Xnew: numpy array
        Array of arrays ( each column is a word vector ).
    """
    X = np.array(word_vectors).T
    Xnew = X.copy()
    for i in range(n):
        Xnew[np.argmax(np.abs(X[:, i])), i] = 0
    return Xnew


def plot_vectors(word_vectors):
    """
    Plot the vectors

    Parameters
    ----------
    word_vectors : list
        List of vectors of a word.


    Returns
    -------

    """
    X = np.array(word_vectors).T
    plt.plot(X)
    plt.show()
    
def get_eig(word_vectors, kernal='k2'):
    """
    Gets the eigenvalues and the eigenvectors  of a kernal

    Parameters
    ----------
    word_vectors : list
        List of vectors of a word.
    keral : str
        'k1' or 'k2'
        --if set to 'k1' : linear kernal.
        --if set to 'k2' : Gaussian kernal.

    Returns
    -------
    lam : numpy array
        eigenvalues
    vec : numpy array
        eigenvectors
    """
    X = np.array(word_vectors).T
    p, n = X.shape
    if kernal=='k1':
        G = (X.T @ X / np.sqrt(p)) / np.sqrt(p)
        lam, vec = np.linalg.eig(G)
    if kernal=='k2':
        f = lambda t: np.exp(-t) # (t-tau)**2 + 2*(t-tau)
        K = f(ed(X.T, X.T)**2 / p)
        P = np.eye(n) - np.ones((n, n)) / n
        lam, vec = np.linalg.eig(P@K@P)
    return lam, vec

def plot_kernal(vec, clusters=None, labels=None, size=(6, 6), annotate=False, annotation=None):
    """
    Plots the first eigenvector of a kernal in function of the second one.
    Parameters
    ----------
    vec : numpy array
        the eigenvectors
    clusters : dict, optional
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])
    labels : list, optional
        list of lables for the samples (used as colors, must be of size vec.shape[0])
    size : tuple
        the size of the figure
    annotate : boolean
        set annotation on points or no
    annotation : list, optional
        list of labels to write besid each point, must be of size vec.shape[0]

    Returns
    -------
    
    """
    fig, ax = plt.subplots(figsize=size)
    if labels is None:
        if clusters is None:
            labels = None
        else:
            labels = np.array([v for k,v in clusters.items()])
    ax.scatter(vec[:, 0], vec[:, 1], c=labels)
    if annotate:
        if annotation is None:
            n = vec.shape[0]
            for i in range(n):
                ax.annotate(i, (vec[:, 0][i], vec[:, 1][i]))
        else:
            for i, txt in enumerate(annotation):
                ax.annotate(txt, (vec[:, 0][i], vec[:, 1][i]))

    
    
def plot_eigval(lam):
    """
    Plots the eigenvalues
    Parameters
    ----------
    lam : numpy array
        the eigenvalues

    Returns
    -------
    
    """
    plt.hist(lam**0.1, 50)
    plt.yscale('log')
    plt.show()
    
def plot_summary(word_vectors, clusters=None):
    """
    Plots a summary (kernal, eigenvalues, blocks, vectors)
    Parameters
    ----------
    word_vectors : list
        list of word vectors
    clusters : dict, optional
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])

    Returns
    -------
    
    """
    X = np.array(word_vectors).T
    fig0, ax0 = plt.subplots()
    ax0.plot(X)
    ax0.set_title('word vectors')
    
    lam1, vec1 = get_eig(word_vectors, 'k1')
    lam2, vec2 = get_eig(word_vectors, 'k2')
    if clusters is None:
        labels = None
    else:
        labels = np.array([v for k,v in clusters.items()])
    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    axs[0, 0].scatter(vec1[:, 0], vec1[:, 1], c=labels)
    axs[0, 0].set_title('v[0], v[1] kernal 1')
    axs[0, 1].scatter(vec2[:, 0], vec2[:, 1], c=labels)
    axs[0, 1].set_title('v[0], v[1] kernal 2')
    axs[1, 0].hist(lam1**0.1, 50)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('eigenvalues kernal 1')
    axs[1, 1].hist(lam2**0.1, 50)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('eigenvalues kernal 2')
    
    if  clusters is None:
        return
    else:
        fig1, ax1 = plt.subplots()
        sorted_vectors, indexes, coms = get_sorted_vectors(word_vectors, clusters)
        sorted_vectors = np.array(sorted_vectors).T
        p, n = sorted_vectors.shape
        f = lambda t: np.exp(-t) 
        K = f(ed(sorted_vectors.T, sorted_vectors.T)**2 / p)
        ax1.imshow(K, cmap='gray')        


def get_sorted_vectors(word_vectors, clusters):
    """
    Gets the vectors sorted by class given by clusters
    Parameters
    ----------
    word_vectors : list
        list of word vectors
    clusters : dict
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])

    Returns
    -------
    sorted_vectors : list
        list of sorted word vectors
    indexes : list
        list of true indexes in dataset of each vector in sorted vectors.
    coms : list
        list that contains lists of nodes in each community.
    """
    sorted_vectors = []
#     balanced = []
    indexes = [0]
    X_bank = word_vectors
    inv_cluster = {v:k for k,v in clusters.items()}
    coms = []
    for k, v in inv_cluster.items():
        coms.append([node for node, val in clusters.items() if val == k])
    for com in coms:
        indexes.append(len(com)+indexes[-1])
        for node in com:
            sorted_vectors.append(X_bank[node])
#     for com in coms[1:]:
#         for i, node in enumerate(com):
#             if i < 500:
#                 balanced.append(X_bank[node])
#     sorted_vectors = np.array(sorted_vectors).T
    return sorted_vectors, indexes, coms

def plot_blocks(word_vectors, clusters):
    """
    Plots the blocks plot 
    Parameters
    ----------
    word_vectors : list
        list of word vectors
    clusters : dict
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])

    Returns
    -------
    
    """
    sorted_vectors, indexes, coms = get_sorted_vectors(word_vectors, clusters)
    sorted_vectors = np.array(sorted_vectors).T
    p, n = sorted_vectors.shape
    f = lambda t: np.exp(-t) 
    K = f(ed(sorted_vectors.T, sorted_vectors.T)**2 / p)
    plt.imshow(K, cmap='gray')
    plt.show()

def sqrtm(c):
    lam, vec = np.linalg.eig(c)
    return np.real(vec) @ np.diag(np.real(np.sqrt(lam))) @ np.real(vec.T)

def generate_gaussian_data(word_vectors, clusters):
    """
    Generates Gaussian data based on the the initial datasets (the same original size)
    Parameters
    ----------
    word_vectors : list
        list of word vectors
    clusters : dict
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])

    Returns
    -------
    X_hat : numpy array
        array of sorted word vectors (each column is a word vector)
    labels : list
        list of labels for each vector in X_hat
    
    """
    sorted_vectors, indexes, coms = get_sorted_vectors(word_vectors, clusters)
    sorted_vectors = np.array(sorted_vectors).T
    X_tmp = []
    labels = []
    for i in range(len(indexes)-1):
        m = np.mean(sorted_vectors[:,indexes[i]:indexes[i+1]], axis=1)
        c = np.cov(sorted_vectors[:,indexes[i]:indexes[i+1]])
        p, n = sorted_vectors[:,indexes[i]:indexes[i+1]].shape
        X_tmp.append(m.reshape(1024, 1) @ np.ones((1, n)) + sqrtm(c) @ np.random.randn(p, n))
        for j in range(indexes[i+1]-indexes[i]):
            labels.append(i)
    X_hat = np.concatenate(X_tmp, axis=1)
    return X_hat, labels


def add_samples(word_vectors, clusters):
    """
    Adds Gaussian vectors to the list of vectors to make a balanced dataset
    Parameters
    ----------
    word_vectors : list
        list of word vectors
    clusters : dict
        dictionnary maps each sample to a cluster (used as colors, must be of size vec.shape[0])

    Returns
    -------
    vectors : list
        list of word vectors
    
    """
    vectors = word_vectors.copy().tolist()
    sorted_vectors, indexes, coms = get_sorted_vectors(word_vectors, clusters)
    lengths = [len(com) for com in coms]
    maxi = np.array(lengths).argmax()
    to_add = [i for i in range(len(lengths)) if i != maxi]
    value_to_add = {i:lengths[maxi]-lengths[i] for i in to_add}
    sorted_vectors = np.array(sorted_vectors).T
    for i in range(len(indexes)-1):
        if i != maxi:
            m = np.mean(sorted_vectors[:,indexes[i]:indexes[i+1]], axis=1)
            c = np.cov(sorted_vectors[:,indexes[i]:indexes[i+1]])
            p, n = sorted_vectors[:,indexes[i]:indexes[i+1]].shape
            n = value_to_add[i]
            X_tmp = m.reshape(1024, 1) @ np.ones((1, n)) + sqrtm(c) @ np.random.randn(p, n)
            for v in X_tmp.T.tolist():
                vectors.append(v)
    return vectors


def get_final_clusters(word=None, k=10, remove_activation=False,  path='/datasets/polysemy/all_vectors.txt', vectors=None, cossim=None):
    vectors_list =[]
    if vectors is None:
        vectors_list = getWordVectors(word, path=path)
    else:
        vectors_list = vectors.copy()
    X=vectors_list.copy()
    n = len(vectors_list)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    if cossim is None:
        sims = getCosSim(word, remove_activation, path, vectors=vectors_list.copy())
    else:
        sims = cossim.copy()
    maxi = max(sims.values())
    mini = min(sims.values())

    for i in sims.keys():
        B[i[0], i[1]] = 100 * ((sims[i]-mini)/(maxi-mini))
        B[i[1], i[0]] = B[i[0], i[1]]
    for i in range(n):
        indexes = B[i].argsort()[-k:][::-1]
        for j in indexes:
            A[i][j] = B[i][j]
    G = nx.from_numpy_matrix(A)

    partition = community_louvain.best_partition(G, random_state=r)

    clusters = partition.copy()
    cv = get_cluster_vector(word, clusters, path, vectors=X)
    vectors_list = [v for k,v in cv.items()]
    cos_sim = dict()
    for ind1, vect1 in enumerate(vectors_list):
        for ind2, vect2 in enumerate(vectors_list[ind1+1:]):
            sec = ind1+ind2+1
            vec1 = vectors_list[ind1]
            vec2 = vectors_list[sec]
            cos_sim[(ind1, sec)] = 1 - spatial.distance.cosine(vec1, vec2) 

    th=1
    coef=1
    best1=-1
    best2=1
    print('starting....')
    while True:
        clusters = partition.copy()

        remove_overlap = dict()
        c_index = 0
        final_clusters = dict()
        for k, v in cos_sim.items():
            if v >= th:
                found = False
                for k1, v1 in remove_overlap.items():
                    if k[0] in v1 or k[1] in v1:
                        v1.append(k[0])
                        v1.append(k[1])
                        found = True
                        break
                if not found:
                    remove_overlap[c_index] = [k[0], k[1]]
                    c_index += 1
        n = len(remove_overlap.keys())
        for i in range(n-1):
            if i in remove_overlap.keys():
                for j in range(i+1, n):
                    if j in remove_overlap.keys():
                        if len(set.intersection(set(remove_overlap[i]), set(remove_overlap[j])))>0:
                            remove_overlap[i] = set.union(set(remove_overlap[i]), set(remove_overlap[j]))
                            remove_overlap.pop(j)
        i = 0
        for k, v in remove_overlap.items():
            final_clusters[i] = set(v)
            i += 1
        for i in range(len(cv.keys())):
            found = False
            for k, v in remove_overlap.items():
                if i in v:
                    found = True
                    break
            if not found:
                final_clusters[len(final_clusters.keys())] = set([i])

        for k, v in clusters.items():
            for k1, v1 in final_clusters.items():
                if v in v1:
                    clusters[k] = k1
                    break

        labels = [v for k, v in clusters.items()]
        if len(set(labels))==1:
            print(th, len(set(labels)))
            print('skipped')
            print()
            th = min(best2 + 0.1/coef, 1) - 0.2/(coef*10)
            coef *= 10
            continue
        score1=metrics.silhouette_score(X, labels, metric='cosine')
        score2=metrics.calinski_harabasz_score(X, labels)
        print(th, len(set(labels)))
        print(score1)
        print(score2)
        print()
        if score2>best1:
            best1 = score2
            best2 = th
            bestcluster = clusters.copy()
            th = th - 0.1/coef
        elif score2==best1:
            best1 = score2
            th = th - 0.1/coef
        else:
            th = min(best2 + 0.1/coef, 1) - 0.1/(coef*10)
            coef *= 10
        if coef > 1000:
            break
    return bestcluster


def get_final_7clusters(word=None, k=10, remove_activation=False,  path='data/SemEval 2010/train/vectors/word_vectors.txt', vectors=None, cossim=None):
    vectors_list =[]
    if vectors is None:
        vectors_list = getWordVectors(word, path=path)
    else:
        vectors_list = vectors.copy()
    X=vectors_list.copy()
    n = len(vectors_list)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    if cossim is None:
        sims = getCosSim(word, remove_activation, path, vectors=vectors_list.copy())
    else:
        sims = cossim.copy()
    maxi = max(sims.values())
    mini = min(sims.values())

    for i in sims.keys():
        B[i[0], i[1]] = 100 * ((sims[i]-mini)/(maxi-mini))
        B[i[1], i[0]] = B[i[0], i[1]]
    for i in range(n):
        indexes = B[i].argsort()[-k:][::-1]
        for j in indexes:
            A[i][j] = B[i][j]
    G = nx.from_numpy_matrix(A)

    partition = community_louvain.best_partition(G, random_state=r)
    nbclusters = len(set(partition.values()))
    print("number of clusters before merge for k=", k, " is: ", len(set(partition.values())))
    clusters = partition.copy()
    cv = get_cluster_vector(word, clusters, path, vectors=X)
    vectors_list = [v for k,v in cv.items()]
    cos_sim = dict()
    for ind1, vect1 in enumerate(vectors_list):
        for ind2, vect2 in enumerate(vectors_list[ind1+1:]):
            sec = ind1+ind2+1
            vec1 = vectors_list[ind1]
            vec2 = vectors_list[sec]
            cos_sim[(ind1, sec)] = 1 - spatial.distance.cosine(vec1, vec2) 

    if nbclusters <= 7:
        clusters7 = True
    else:
        clusters7 = False
    bestcluster = clusters
    th = 1
    best = 2
    print('merge is starting...')
    while not clusters7:
        clusters = partition.copy()
        remove_overlap = dict()
        c_index = 0
        final_clusters = dict()
        for k, v in cos_sim.items():
            if v >= th:
                found = False
                for k1, v1 in remove_overlap.items():
                    if k[0] in v1 or k[1] in v1:
                        v1.append(k[0])
                        v1.append(k[1])
                        found = True
                        break
                if not found:
                    remove_overlap[c_index] = [k[0], k[1]]
                    c_index += 1
        n = len(remove_overlap.keys())
        for i in range(n-1):
            if i in remove_overlap.keys():
                for j in range(i+1, n):
                    if j in remove_overlap.keys():
                        if len(set.intersection(set(remove_overlap[i]), set(remove_overlap[j])))>0:
                            remove_overlap[i] = set.union(set(remove_overlap[i]), set(remove_overlap[j]))
                            remove_overlap.pop(j)
        i = 0
        for k, v in remove_overlap.items():
            final_clusters[i] = set(v)
            i += 1
        for i in range(len(cv.keys())):
            found = False
            for k, v in remove_overlap.items():
                if i in v:
                    found = True
                    break
            if not found:
                final_clusters[len(final_clusters.keys())] = set([i])

        for k, v in clusters.items():
            for k1, v1 in final_clusters.items():
                if v in v1:
                    clusters[k] = k1
                    break
                    

        labels = [v for k, v in clusters.items()]
        if th == best:
            print('Found!! for threshold=',th,' , we have ',len(set(labels)), ' clusters.' )
            bestcluster = clusters.copy()
            clusters7 = True
        if len(set(labels)) < 7:
            print('for threshold=',th,' , we have ',len(set(labels)), ' clusters.' )
            if best == 2:
                best = th
            else:
                th=best
        else:
            print('for threshold=',th,' , we have ',len(set(labels)), ' clusters.' )
            best = th
            th -= 0.001
        
    return bestcluster
