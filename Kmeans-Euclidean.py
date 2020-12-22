import os
import sys
import pickle
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from utils import time_wrap, txt_csr, load_tsne
    
@time_wrap
def kmeans(csr_mat, n_c=5, name=None):
    print("C={}, Start Kmeans Clustering".format(n_c))
    kmeans = KMeans(n_clusters=n_c).fit(csr_mat)
    print("C={}. Complete Kmeans Clustering".format(n_c))
    overall_d = kmeans.inertia_**0.5
    print("overall distance of C-{}-{}-kmeans-euclidean: {:.2f}".format(n_c, name, overall_d))
    return kmeans.labels_

def visual(tsne, labels, n_c, name="Music"):
    label_count = Counter(labels).most_common()
    del_label = label_count[0][0]

    labels = np.array(labels)
    keep = labels != del_label
    temp_tsne = tsne[keep]
    temp_labels = labels[keep]

    plt.style.use('seaborn-darkgrid')
    plt.scatter(temp_tsne[:,0], temp_tsne[:,1], c=temp_labels, cmap='viridis',s=2)
    plt.savefig(name+"-Kmeans-Eud-C-{}.jpg".format(n_c))

if __name__=='__main__':
    music_txt = os.getcwd()+'/music_data.txt'
    music_csr = txt_csr(music_txt)
    
    music_tsne = load_tsne('pickle/music_tsne.pickle')
    for n_c in [5, 10, 20]:
        labels = kmeans(music_csr, n_c=n_c, name='music')
        visual(music_tsne, labels, n_c=n_c, name="Music")