import time
import pickle
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from scipy.sparse import coo_matrix, csr_matrix

def time_wrap(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{} process time: {:2f}".format(func.__name__, end-start))
        return result
    return wrapper

@time_wrap
def txt_csr(file_path):
    """
    convert txt file from HW_1 to csr matrix

    params:
    file_path --> path of txt file
    
    return:
    A sparse metrix in csr format
    """

    with open(file_path, 'r') as file:
        products = defaultdict(lambda:0)

        user_idx = 0
        product_idx = 0
        row , col, ones = [], [], []
        lines = file.readlines()
        for line in lines[:-1]:
            user, *items = [t.strip() for t in line.split(',')]
            for item in items:
                if item not in products:
                    products[item] = product_idx
                    product_idx += 1

                row.append(user_idx)
                col.append(products[item])
                ones.append(1)
            user_idx += 1

    coo_mat = coo_matrix((ones, (row, col)), dtype=np.uint8, shape=(user_idx, product_idx))
    csr_mat = coo_mat.tocsr()
    return csr_mat

@time_wrap
def tsne(csr_mat, save=False, save_name=None):
    print("Start TSNE")
    tsne = TSNE(n_iter=250).fit_transform(csr_mat)
    print("Complete TSNE")
    if save:
        assert isinstance(save_name, str), "save_name has to be a string"
        with open(save_name+'.pickle', 'wb') as file_save:
            pickle.dump(tsne, file_save)
    return tsne

def load_tsne(file_path):
    with open(file_path, 'rb') as pick_f:
        tsne = pickle.load(pick_f)
    return tsne