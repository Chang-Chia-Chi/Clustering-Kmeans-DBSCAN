import os
import sys
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict, Counter, deque
from scipy.sparse import coo_matrix, csr_matrix
from utils import time_wrap, txt_csr, load_tsne

class DBSACN_Jaccard:
    """
    Run dbscan-jaccard algorithm with sparse matrix

    Reference: 
    https://github.com/aahoo/dbscan/blob/master/dbscan.py
    https://www.youtube.com/watch?v=DBpTY6J3ttM&list=PLgQKp4YLJUdy5fm8ifluSl8J96RwsbHAd&index=12
    """
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts

    def non_zero(self, csr_mat, verbos=False, save=False, save_name=None):
        """
        Build a dictionary of non_zero entries: users.

        keys --> frozenset of col indices of nonzero entries for each user
        values --> users (row index)
        """
        if verbos:
            print("Build non_zero dictionary")
            t_start = time()

        p_start = 0
        non_zero = defaultdict(list)
        for user, p_end in enumerate(csr_mat.indptr[1:]):
            non_zero[frozenset(csr_mat.indices[p_start:p_end])].append(user)
            p_start = p_end
        
        if save:
            assert isinstance(save_name, str), "save_name has to be a string"
            with open(save_name+'.pickle', 'wb') as file_save:
                pickle.dump(non_zero, file_save)

        if verbos:
            print("Complete non_zero dictionary in {:.2f} s".format((time()-t_start)))
        return non_zero

    def nn_graph(self, csr_mat, non_zero=None, verbos=False, save=False, save_name=None):
        """
        Build nearest neighbor graph for each user
        """
        if non_zero is None:
            non_zero = self.non_zero(csr_mat)
        
        if verbos:
            print("Build nearest neighbor graph")
            t_start = time()
        
        # initialze graph
        graph = {i: [] for i in range(len(non_zero))}
        # sorting product combination based on number of items
        records = list(non_zero.keys())
        sort_buy = Counter({i: len(key) for i, key in enumerate(records)}).most_common()

        # build nearest neighbor graph,
        # because number of products bought have been sorted,
        # we could assume the best case that j_buy is 
        # subset of i_buy and check eps prior.
        # if (i_buy - j_buy)/i_buy > eps, we can break inner loop
        # because it's impossible for rest items to meet criteria
        eps = self.eps
        for k, (i, i_buy) in enumerate(sort_buy):
            for j, j_buy in sort_buy[k+1:]:
                if j_buy < (1 - eps)*i_buy:
                    break
            
                intersect = len(records[i] & records[j])
                if 1. - intersect/(i_buy+j_buy-intersect) <= eps:
                    graph[i].append(j)
                    graph[j].append(i)
        if save:
            assert isinstance(save_name, str), "save_name has to be a string"
            with open(save_name+'.pickle', 'wb') as file_save:
                pickle.dump(graph, file_save)

        if verbos:
            print("Complete building nearest neighbor graph in {:.2f} s".format(time()-t_start))

        return graph

    def fit(self, csr_mat, graph=None, non_zero=None, verbos=False):
        """
        DBSCAN clustering with sparse matrix
        """
        if non_zero is None:
            non_zero = self.non_zero(csr_mat)
        records = list(non_zero.keys())

        if graph is None:
            graph = self.nn_graph(csr_mat, non_zero=non_zero, verbos=verbos)
        if verbos:
            print("Start DBSCAN clustering")
            t_start = time()

        visited = set()
        clusters = [set()]
        for idx, neighbors in graph.items():
            if idx not in visited:
                visited.add(idx)
                n_neighbors = len([user for i in neighbors for user in non_zero[records[i]]])
                if n_neighbors + len(non_zero[records[idx]]) <= self.minPts:
                    clusters[0].add(idx)
                else:
                    clusters.append(set([idx]))
                    queue = deque(neighbors)
                    while queue:
                        p = queue.popleft()
                        if p not in visited:
                            visited.add(p)
                            if len(graph[p])+len(non_zero[records[p]]) >= self.minPts:
                                queue.extend(graph[p])
                        clusters[-1].add(p)
        clusters = [[user for i in c for user in non_zero[records[i]]] for c in clusters]
        if verbos:
            print("Complete DBSCAN clustering in {:.2f} s".format(time()-t_start))
            print("Total number of clusters: {}".format(len(clusters)))
            print("Largest cluster size: {}".format(max(map(len, clusters[1:]))))
            print("Total number of noise: {}".format(len(clusters[0])))

        user_labels = {user: i
            for i, c in enumerate(clusters) for user in c
        }
        n_users = len(user_labels)
        labels = [user_labels[i] for i in range(n_users)]
        return clusters, labels

    def overall_dist(self, csr_mat, clusters, verbos=False, name=None):
        """
        Compute overall euclidean distance samples to corresponded cluster center
        """
        if verbos:
            print("Calculate overall distance")
            t_start = time()

        dist = 0
        n_items = max(csr_mat.indices) + 1
        for c in clusters[1:]:
            user_buy = defaultdict(list)
            item_buy = np.zeros(n_items)
            for user in c:
                i_start, i_end = csr_mat.indptr[user], csr_mat.indptr[user+1]
                items = csr_mat.indices[i_start:i_end]
                user_buy[user] = items
                for item in items:
                    item_buy[item] += 1

            n_users = len(c)
            centroid = item_buy/n_users
            for items in user_buy.values():
                diff = centroid.copy()
                diff[items] -= 1
                dist += np.sum(diff**2)

        dist = np.sqrt(dist)
        if verbos:
            print("Overall distance of {}-dbscan-jaccard is {:.2f}".format(name, dist))
            print("Complete overall distance calculation in {:.2f} s".format(time()-t_start))

        return dist

    def load_dict(self, file_path):
        with open(file_path, 'rb') as file_read:
            dictionary = pickle.load(file_read)
        return dictionary
    
    def visual(self, tsne, labels, name="Music"):
        label_count = Counter(labels).most_common()
        del_label = label_count[0][0]
    
        labels = np.array(labels)
        keep = labels != del_label
        temp_tsne = tsne[keep]
        temp_labels = labels[keep]

        plt.style.use('seaborn-darkgrid')
        plt.scatter(temp_tsne[:,0], temp_tsne[:,1], c=temp_labels, cmap='viridis', s=2)
        plt.savefig(name+"-DBSCAN-Jaccard.jpg")

if __name__=='__main__':
    music_txt = os.getcwd()+'/music_data.txt'
    music_csr = txt_csr(music_txt)
    music_tsne = load_tsne("pickle/music_tsne.pickle")

    db = DBSACN_Jaccard(0.5, 5)
    # If you want run nearest neighbor graph from scratch, use line below and comment the second line (it took hours to run)
    # graph = db.nn_graph(music_csr, non_zero=non_zero, verbos=True, save=True, save_name="music_graph")
    graph = db.load_dict("pickle/music_graph.pickle") 
    non_zero = db.non_zero(music_csr, verbos=True, save=False, save_name="music_nonzero")
    clusters, labels = db.fit(music_csr, graph=graph, non_zero=non_zero, verbos=True)
    dist = db.overall_dist(music_csr, clusters, verbos=True, name='music')
    db.visual(music_tsne, labels, name="Music")