# -*- coding: utf-8 -*-

import re
import string
import operator
import os
import numpy as np

class Dataset:
    def __init__(self):
        '''constructor
        '''
        self.data = []
        self.classes = []   
        self.label = []
        self.bag_of_word = {}
        self.dicts = []
        self.counts = []
        self.tf_idf = []
        self.topk_tf_idf = []
        self.sparse_vector = []

    def read_data(self, folder_path):
        '''
        folder_path: class folder_path, each txt file in the folder named from 0.txt to the files_num.txt
        return a list(self.data) which contents each file object 
        '''
        for directory in os.listdir(folder_path): #folder
            if os.path.isdir(os.path.join(folder_path, directory)):
                self.classes.append(directory)
        for item in self.classes:
            path = os.path.join(folder_path, item)
            len0 = len(self.data)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    file_path = os.path.join(path, file)
                    content = open(file_path, 'r').read()
                    self.data.append(content)
            len1 = len(self.data)
            item_label = [self.classes.index(item)] * (len1 - len0)
            self.label += item_label


    def getNgrams(self, ngram):
        '''
        top_k: the top k high frequence ngrams
        ngram: define unigram/bigram/trigram
        return 
        '''
        value = 0
        for content in self.data:
            content = content.split(' ')[1:]
            tmp_count = {}
            for i in range(len(content)-ngram+1):
                ngramTemp = " ".join(content[i:i+ngram])#.encode('utf-8')
                if ngramTemp not in self.bag_of_word:
                    self.bag_of_word[ngramTemp] = value
                    value += 1
                if ngramTemp not in tmp_count:
                    tmp_count[ngramTemp] = 0 
                tmp_count[ngramTemp] += 1
            self.counts.append(tmp_count)

    def cal_tf_idf(self):
        num_fileWithNgram = {}
        for count in self.counts:
            for key in count.keys():
                if key not in num_fileWithNgram:
                    num_fileWithNgram[key] = 0
                num_fileWithNgram[key] += 1

        idf = {}
        files_num = len(self.data)
        for key in num_fileWithNgram.keys():
            idf_val = np.log(float(files_num)/float((1+num_fileWithNgram[key])))
            idf[key] = idf_val

        for file_ngram in self.counts:
            tf_idf_file = {}
            total_freq = 0
            for val in file_ngram.values():
                total_freq += val
            for key, val in file_ngram.items():
                tf = float(val)/float(total_freq)
                tf_idf_file[key] = tf * idf[key]
            self.tf_idf.append(tf_idf_file)

    def get_top_k(self, top_k):
        '''
        top_k: the top_k ngrams 
        '''
        for i in self.tf_idf:
            topk_ifidf = sorted(i.items(),key = lambda item:item[1], reverse = True)[:top_k]
            self.topk_tf_idf.append(topk_ifidf)


    def get_sparse_vector(self):
        for file in self.topk_tf_idf:
            tmp_sparse_vector = [0] * len(self.bag_of_word)
            for tuple_item in file:
                tmp_sparse_vector[self.bag_of_word[tuple_item[0]]] = tuple_item[1]
                # print(self.bag_of_word[tuple_item[0]])
            self.sparse_vector.append(tmp_sparse_vector) # append list to get matrix

    def split_train_valid(self):
        idx = np.random.permutation(len(self.label))
        x, y = np.array(self.sparse_vector)[idx].tolist(), np.array(self.label)[idx].tolist()
        x = self.normalize(x)
        idx = int(len(x)*0.9)
        self.train_x, self.train_y = x[:idx], y[:idx]
        self.valid_x, self.valid_y = x[idx:], y[idx:]

    def normalize(self, feature_sparse_vecs):
        normalize_vec = []

        for feature_sparse_vec in feature_sparse_vecs:
            total = 0
            for item in feature_sparse_vec:
                total += item
            normalize_vec_tmp = [float(norm_item)/float(total) for norm_item in feature_sparse_vec]
            normalize_vec.append(normalize_vec_tmp)

        return normalize_vec


            




















        




