import numpy as np
import numpy.random as npr
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from math import ceil
from gensim.models import Word2Vec

'''
featur1     is the first feature matrix
alpha1      is the weight for the first feature matrix, 0 <= alpha1 <= 1
featur2     is the second feature matrix
alpha2      is the weight for the second feature matrix, 0 <= alpha2 <= 1, 0 <= alpha1+alpha2 <= 1
Net         is the last feature matrix, which describes the relations among instances, its weight is 1-alpha1-alpha2
beta        is the small value threshold
num_paths   is the number of feature walks to start at each instance
path_length is the length of the feature walk started at each instance
dim         is the dimension of embedding representations
win_size    is the window size of skipgram model
'''

class featurewalk:
    def __init__(self, featur1, alpha1, featur2, alpha2, Net, beta, num_paths, path_length, dim, win_size):
        self.n = featur1.shape[0]  # number of instance
        if alpha1+alpha2 < 1:  # Embed Network
            Net = csc_matrix(Net)
            self.path_list_Net = []
            self.idx = []
            for i in range(self.n):
                self.path_list_Net.append(Net.getcol(i).nnz)
                self.idx.append(Net.getcol(i).indices)

        if alpha1 > 0:  # Embed Feature 1
            self.featur1 = featur1
        if alpha2 > 0:  # Embed Feature 2
            self.featur2 = featur2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.num_paths = num_paths
        self.path_length = path_length
        self.d = dim
        self.win_size = win_size

    def throughfeatur(self):  # Walk through Feature
        sentlist = []
        for i in self.allidx:
            for j in range(ceil(self.path_list[i] * self.weight)):
                current = i
                feai = self.idx[i][alias_draw(self.JListrow[i], self.qListrow[i])]  # feature idx
                sentence = [self.idxListcol[feai][alias_draw(self.JListcol[feai], self.qListcol[feai])], i]
                for tmp in range(self.path_length - 2):
                    feai = self.idx[current][alias_draw(self.JListrow[current], self.qListrow[current])]  # feature idx
                    current = self.idxListcol[feai][alias_draw(self.JListcol[feai], self.qListcol[feai])]  # ahead
                    sentence.append(current)
                sentlist.append([str(word) for word in sentence])
        return sentlist

    def function(self):
        max_memory = 233000000.0/self.path_length # 235000000 For 16GB
        bulidflag = True
        sentenceList = []  # All the walks will be here

        # Embed Net first, such that we could del Net earlier
        alpha = float(1 - self.alpha1 - self.alpha2)
        if alpha > 0:  # Embed Network
            allidx = np.nonzero(self.path_list_Net)[0]
            if len(allidx) != self.n:  # initialize with Network
                for i in np.where(np.asarray(self.path_list_Net) == 0)[0]:
                    sentenceList.append([str(i)])
            splitnum = int(ceil(self.n * self.num_paths * alpha / max_memory))  # split because of memory limit
            for blocki in range(splitnum):
                weight = min(max_memory, alpha * self.num_paths * self.n - max_memory*blocki) / np.sum(self.path_list_Net)
                # path_list = [ceil(i * weight) for i in self.path_list_Net]
                for i in allidx:
                    for j in range(ceil(self.path_list_Net[i] * weight)):
                        sentence = [npr.choice(self.idx[i]), i]
                        for tmp in range(self.path_length - 2):
                            sentence.append(npr.choice(self.idx[sentence[-1]]))
                        sentenceList.append([str(word) for word in sentence])
                if splitnum >= 2 and blocki != splitnum - 1:  # If # Not enough memory, it is splited, and not the last iteration
                    if bulidflag:
                        model = Word2Vec(sentenceList, size=self.d, window=self.win_size, min_count=0)
                        bulidflag = False
                    else:
                        model.build_vocab(sentenceList, update=True)
                        model.train(sentenceList, total_examples=len(sentenceList), epochs=model.iter)
                    sentenceList = []
            del self.path_list_Net


        for alphaidx in range(2):  # Embed Feature
            memory1 = len(sentenceList)
            if alphaidx == 0:
                alpha = float(self.alpha1)
                if alpha > 0:
                    featur = normalize(csc_matrix(self.featur1), norm='l2')
                    del self.featur1
            if alphaidx == 1:
                alpha = float(self.alpha2)
                if alpha > 0:
                    featur = normalize(csc_matrix(self.featur2), norm='l2')
                    del self.featur2

            if alpha > 0:
                if self.beta > 0:
                    featur = featur.multiply(featur > self.beta * (featur.sum() / featur.nnz))  # remove small elements
                featur = featur[:, np.where((featur != 0).sum(axis=0) > 1)[1]].T

                self.path_list = []
                for i in range(self.n):
                    self.path_list.append(featur.getcol(i).nnz)
                sumpath = np.sum(self.path_list)

                featur = normalize(featur, norm='l1', axis=0)
                self.qListrow = []  # for each instance
                self.JListrow = []
                self.idx = []
                for ni in range(self.n):  # for each instance
                    coli = featur.getcol(ni)
                    J, q = alias_setup(coli.data)
                    self.JListrow.append(J)
                    self.qListrow.append(q)
                    self.idx.append(coli.indices)
                self.qListcol = []  # for each feature
                self.JListcol = []
                self.idxListcol = []
                featur = normalize(featur.T, norm='l1', axis=0)
                for ni in range(featur.shape[1]):  # for each feature
                    coli = featur.getcol(ni)
                    J, q = alias_setup(coli.data)
                    self.JListcol.append(J)
                    self.qListcol.append(q)
                    self.idxListcol.append(coli.indices)
                del featur, coli, J, q

                self.allidx = np.nonzero(self.path_list)[0]
                if self.alpha1 + self.alpha2 == 1 and len(self.allidx) != self.n:
                    for i in np.where(np.asarray(self.path_list) == 0)[0]:
                        sentenceList.append([str(i)])

                memory2 = self.n * self.num_paths * alpha
                if memory2 < max_memory - memory1:  # Enough Memory
                    self.weight = memory2 / sumpath
                    sentenceList.extend(self.throughfeatur())
                else:  # Not Enough Memory
                    self.weight = (max_memory - memory1) / sumpath
                    sentenceList.extend(self.throughfeatur())
                    if bulidflag:
                        model = Word2Vec(sentenceList, size=self.d, window=self.win_size, min_count=0)
                        bulidflag = False
                    else:
                        model.build_vocab(sentenceList, update=True)
                        model.train(sentenceList, total_examples=len(sentenceList), epochs=model.iter)
                    sentenceList = []
                    splitnum = int(ceil((memory2 + memory1) / max_memory - 1))
                    for blocki in range(splitnum):
                        self.weight = min(max_memory, memory2 + memory1 - max_memory * (blocki + 1)) / sumpath
                        sentenceList.extend(self.throughfeatur())
                        if blocki != splitnum - 1:
                            model.build_vocab(sentenceList, update=True)
                            model.train(sentenceList, total_examples=len(sentenceList), epochs=model.iter)
                            sentenceList = []
                del self.path_list, self.JListrow, self.qListrow, self.idx, self.JListcol, self.qListcol, self.idxListcol, self.allidx
        if bulidflag:
            model = Word2Vec(sentenceList, size=self.d, window=self.win_size, min_count=0)
        else:
            model.build_vocab(sentenceList, update=True)
            model.train(sentenceList, total_examples=len(sentenceList), epochs=model.iter)
        del sentenceList
        H = np.zeros((self.n, self.d))
        for i in range(self.n):
            H[i] = model.wv[str(i)]
        return H


#  Compute utility lists for non-uniform sampling from discrete distributions.
#  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]