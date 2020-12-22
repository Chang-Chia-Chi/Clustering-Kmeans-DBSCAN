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

class Kmeans_Jaccard:
    """
    Run kmeans-jaccard algorithm with sparse matrix
    """
    def __init__(self, n_clusters, max_iter=300):
        self.n_c = n_clusters
        self.max_iter = max_iter

    def fit(self, csr_mat, verbos=False):
        """
        Kmeans-Jaccard clustering with sparse matrix
        """
        n_users = len(csr_mat.indptr)-1
        n_items = max(csr_mat.indices) + 1
        user_buy = self.user_record(csr_mat, verbos=verbos)

        # initialize center points
        cen_idx = np.random.randint(0, n_users, self.n_c)
        centroids = {c: user_buy[idx]  for c, idx in enumerate(cen_idx)}
        curr_clus = {c: set() for c in range(self.n_c)}
        prev_clus = {c: set() for c in range(self.n_c)}

        if verbos:
            print("Start Kmeans-Jaccard clustering")
            t_start = time()

        iter_num = 0
        while iter_num < self.max_iter:
            # find closest centroid for each user
            print("Iter {} Start".format(iter_num))
            t_s = time()
            for i in range(n_users):
                closest = None
                min_dist = float('inf')
                for c_i in range(self.n_c):
                    dist = self.jaccard(user_buy, centroids, i, c_i)
                    if dist < min_dist:
                        closest = c_i
                        min_dist = dist

                # if minimum distance equals to 1, set to cluster randomly  
                if min_dist == 1.:
                    closest = random.randint(0, self.n_c-1)
                curr_clus[closest].add(i)

            # if converge, break loop
            if all([curr_clus[i]==prev_clus[i] for i in range(self.n_c)]):
                break
            
            iter_num += 1
            centroids = self.update_centroid(curr_clus, user_buy)
            prev_clus = curr_clus
            curr_clus = {c: set() for c in range(self.n_c)}
            print("Iter {} Complete in {:.2f} s".format(iter_num-1, t_s-time()))
        
        user_labels = {user: i
            for i, c in prev_clus.items() for user in c
        }
        n_users = len(user_labels)
        labels = [user_labels[i] for i in range(n_users)]

        if verbos:
            print("Complete Kmeans-Jaccard clustering in {:.2f} s".format(time()-t_start))
            print("Cluster size: {}".format(list(map(len, prev_clus.values()))))

        return prev_clus, labels

    def jaccard(self, user_buy, centroids, user, center):
        """
        compute jaccard distance between user and center
        """
        intersection = len(user_buy[user][0] & centroids[center][0])
        union = user_buy[user][1] + centroids[center][1] - intersection
        return 1 - intersection/union

    def user_record(self, csr_mat, verbos=False):
        """
        Create user transaction record dictionary:

        keys --> user id
        values --> tuple of (items idx in set each user buy, # of items)
        """
        if verbos:
            print("Start building user transaction record dictionary")
            t_start = time()

        p_start = 0
        user_buy = {}
        for user, p_end in enumerate(csr_mat.indptr[1:]):
            user_buy[user] = (set(csr_mat.indices[p_start:p_end]), p_end - p_start)
            p_start = p_end
        
        if verbos:
            print("Complete dictionary building in {:.2f} s".format(time()-t_start))
        return user_buy

    def update_centroid(self, curr_clus, user_buy):
        """
        Compute new centroids from result of clustering

        step 1.: compute average coordinates as temporary centroid 
                    by counting frequeny of each item and divided by # of user in cluster
        step 2.: choose user with minimum L2 norm to tempoorary centroid as
                    new representative
        step 3.: repeat step 1. & 2. for all clusters
        """
        new_centroids = {c: None  for c in range(self.n_c)}
        for c, users in curr_clus.items():
            n_users = len(users)
            clus_rec = []
            user_list = list(users)

            # get list of all products bought in the cluster and
            # compute freq of each product and temporary centroid
            for user in user_list:
                user_rec = list(user_buy[user][0])
                clus_rec += user_rec
            rec_count = Counter(clus_rec).items()
            item_id = {item: i for i, (item, _) in enumerate(rec_count)}
            temp_cen = np.array([count/n_users for _, count in rec_count])

            # find user obtain minimum L2 norm to temporary centroid as representative
            rep = []
            min_dist = float('inf')
            for user in user_list:
                user_rec = [item_id[item] for item in user_buy[user][0]]
                diff = temp_cen.copy()
                diff[user_rec] -= 1
                dist = np.sum(np.square(diff))
                if dist == min_dist:
                    rep.append(user)
                if dist < min_dist:
                    rep_user = [user]
                    min_dist = dist

            rep_user = random.choice(rep)
            new_centroids[c] = user_buy[rep_user]
        return new_centroids

    def overall_dist(self, csr_mat, clusters, verbos=False, name=None):
        """
        Compute overall euclidean distance of samples to corresponded cluster center
        """
        if verbos:
            print("Calculate overall distance")
            t_start = time()

        dist = 0
        n_items = max(csr_mat.indices) + 1
        for users in clusters.values():
            user_buy = defaultdict(list)
            item_buy = np.zeros(n_items)
            for user in users:
                i_start, i_end = csr_mat.indptr[user], csr_mat.indptr[user+1]
                items = csr_mat.indices[i_start:i_end]
                user_buy[user] = items
                for item in items:
                    item_buy[item] += 1

            n_users = len(users)
            centroid = item_buy/n_users
            for items in user_buy.values():
                diff = centroid.copy()
                diff[items] -= 1
                dist += np.sum(diff**2)

        dist = np.sqrt(dist)
        if verbos:
            print("Overall distance of C-{}-{}-kmeans-jaccard is {:.2f}".format(self.n_c, name, dist))
            print("Complete overall distance calculation in {:.2f} s".format(time()-t_start))

        return dist
    
    def visual(self, tsne, labels, name="Music"):
        label_count = Counter(labels).most_common()
        del_label = label_count[0][0]

        labels = np.array(labels)
        keep = labels != del_label
        temp_tsne = tsne[keep]
        temp_labels = labels[keep]

        plt.style.use('seaborn-darkgrid')
        plt.scatter(temp_tsne[:,0], temp_tsne[:,1], c=temp_labels, cmap='viridis', s=2)
        plt.savefig(name+"-Kmeans-Jaccard-C-{}.jpg".format(self.n_c))

if __name__ == '__main__':
    music_txt = os.getcwd()+'/music_data.txt'
    music_csr = txt_csr(music_txt)

    music_tsne = load_tsne('pickle/music_tsne.pickle')
    kmeans_jac = Kmeans_Jaccard(10)
    clusters, labels = kmeans_jac.fit(music_csr, verbos=True)
    dist = kmeans_jac.overall_dist(music_csr, clusters, verbos=True, name='music')
    kmeans_jac.visual(music_tsne, labels, name="Music")