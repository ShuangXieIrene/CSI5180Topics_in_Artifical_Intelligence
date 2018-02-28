# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
from sklearn.cross_validation import KFold
import numpy as np
from optparse import OptionParser
import sys
import os
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.externals import joblib
import pickle
import os
import errno




print('Loading data:')
#Load train and validation set 
def read_data(folder_path):
    '''
    folder_path: class folder_path, each txt file in the folder named from 0.txt to the files_num.txt
    return a list(self.data) which contents each file object 
    '''
    classes = []
    data = []
    label = []
    for directory in os.listdir(folder_path): #folder
      if os.path.isdir(os.path.join(folder_path, directory)):
        classes.append(directory)
    for item in classes:
      path = os.path.join(folder_path, item)
      len0 = len(data)
      for file in os.listdir(path):
        # if file.endswith('.txt'):
        #   file_path = os.path.join(path, file)
        #   content = open(file_path, 'r').read()
        #   data.append(content)
        if not file.startswith('.'):
          file_path = os.path.join(path, file)
          content = open(file_path, 'r').read()
          data.append(content)
      len1 = len(data)
      item_label = [classes.index(item)] * (len1 - len0)
      label += item_label
    return classes, data, label


# target_names, data, y = read_data(folder_path='../data_10fold')
target_names, data, y = read_data(folder_path='./uri_0.3_data')
print('y', len(y))
data_train = []
y_train = []
data_valid = []
y_valid = []

kf = KFold(len(y), n_folds=10, shuffle=True)
print('kf:', kf)
# print('type:', data[0])
for train_indexs, test_indexs in kf:
    print('train_index', train_indexs)
    print('test_index', test_indexs)
    for train_index in train_indexs:
        data_train.append(data[train_index])
        y_train.append(y[train_index])
    for test_index in test_indexs:
        data_valid.append(data[test_index])
        y_valid.append(y[test_index])


    
    
    # data_train, y_train = data[train_index], y[train_index] 
    # data_valid, y_valid = data[test_index], y[test_index]
    # data_train = data.iloc[train_index]['text'].values
    # y_train = y.iloc[train_index]['class'].values

    # data_valid = data.iloc[test_index]['text'].values
    # y_valid = y.iloc[test_index]['class'].values
# target_names, data_train, y_train = read_data(folder_path='../data/train')
# target_names, data_valid, y_valid = read_data(folder_path='../data/validation')


print("Extracting Tfidf features from the data using a sparse vectorizer")
ngram_range=(1,5)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, smooth_idf=True, ngram_range=ngram_range)
print (vectorizer)



X_train = vectorizer.fit_transform(data_train)
X_valid = vectorizer.transform(data_valid)


# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()




select_chi2=10000
print("Extracting %d best features by a chi-squared test" %
          select_chi2)
if select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          select_chi2)
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_valid = ch2.transform(X_valid)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf, name=None):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_valid)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_valid, pred)
    print("accuracy:   %0.3f" % score)

    # if hasattr(clf, 'coef_'):
    #     print("dimensionality: %d" % clf.coef_.shape[1])
    #     print("density: %f" % density(clf.coef_))

    #     print_top10 = True
    #     if print_top10 and feature_names is not None:
    #         print("top 10 keywords per class:")
    #         for i, label in enumerate(target_names):
    #             top10 = np.argsort(clf.coef_[i])[-10:]
    #             print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
    #     print()


    print_report = True
    if print_report:
        print("classification report:")
        print(metrics.classification_report(y_valid, pred,
                                            target_names=target_names))

    print_cm = True
    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_valid, pred))

    print()
    clf_descr = str(clf).split('(')[0]

    print('save model file into ./result/{name}.pkl'.format(name=name))
    joblib.dump(clf, './result/{name}.pkl'.format(name=name)) 
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,name=name))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3),name='LinearSVC'))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty), name='SGDClassifier'))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet"), name='Elastic-Net'))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid(),name='NearestCentroid'))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01),name='MultinomialNB'))
results.append(benchmark(BernoulliNB(alpha=.01),name='BernoulliNB'))



# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# # The smaller C, the stronger the regularization.
# # The more regularization, the more sparsity.
# results.append(benchmark(Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
#                                                   tol=1e-3))),
#   ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
# training_time = np.array(training_time) / 25.316 #25.316 is the max training time of RF classifier using ngram features
test_time = np.array(test_time) / np.max(test_time)
# test_time = np.array(test_time) / 11.521  #11.521 is the max test time of KNN classifier using ngram features

# for i, result in enumerate(clf):
#     filename = './result/' + clf_names[i] + '.pkl'
#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))

#     with open(filename, 'wb') as f:
#         pickle.dump(result, f)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)



for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
