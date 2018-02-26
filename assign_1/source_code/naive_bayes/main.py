# -*- coding: utf-8 -*-

from dataset import Dataset
from bayesclassifer import TFNaiveBayes
import tensorflow as tf

#preprocess data
dataset = Dataset()
dataset.read_data(folder_path= '../data')
dataset.getNgrams(ngram = 1)
dataset.cal_tf_idf()
dataset.get_top_k(3000)
dataset.get_sparse_vector()

dataset.split_train_valid()

bayesclassifer = TFNaiveBayes()

bayesclassifer.fit_NormalDistribution(dataset.train_x, dataset.train_y)
s = tf.InteractiveSession()
y_predict = s.run(bayesclassifer.predict(dataset.valid_x))

print(y_predict)