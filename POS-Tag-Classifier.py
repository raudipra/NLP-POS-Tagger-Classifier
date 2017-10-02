#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:47:41 2017

@author: raudi candra
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy

X = []
y = []
f = open('en-ud-train.conllu', 'r')
for line in f:
  sentence = line.split("\t")
  if sentence[0].isdigit():
    X.append( {
      'word': sentence[1],
      'is_first': sentence[0] == '1',
      'is_last': sentence[0] == len(sentence) - 1,
      'is_capitalized': sentence[1][0].upper() == sentence[1][0],
      'is_all_caps': sentence[1].upper() == sentence[1],
      'is_all_lower': sentence[1].lower() == sentence[1],
      'prefix-1': sentence[1][0],
      'prefix-2': sentence[1][:2],
      'prefix-3': sentence[1][:3],
      'suffix-1': sentence[1][-1],
      'suffix-2': sentence[1][-2:],
      'suffix-3': sentence[1][-3:],
      'prev_tag': '' if sentence[0] == '1' else y[len(X)-1],
      'prev_word': '' if sentence[0] == '1' else X[len(X)-1]['word'],
      'has_hyphen': '-' in sentence[1],
      'is_numeric': sentence[1].isdigit(),
      'capitals_inside': sentence[1][1:].lower() != sentence[1][1:] }
    )
    # X.append( [
    #   sentence[1],
    #   sentence[0] == '1',
    #   sentence[0] == len(sentence) - 1,
    #   sentence[1][0].upper() == sentence[1][0],
    #   sentence[1].upper() == sentence[1],
    #   sentence[1].lower() == sentence[1],
    #   sentence[1][0],
    #   sentence[1][:2],
    #   sentence[1][:3],
    #   sentence[1][-1],
    #   sentence[1][-2:],
    #   sentence[1][-3:],
    #   '' if sentence[0] == '1' else y[len(X)-1],
    #   '' if sentence[0] == '1' else X[len(X)-1][0],
    #   '-' in sentence[1],
    #   sentence[1].isdigit(),
    #   sentence[1][1:].lower() != sentence[1][1:] ]
    # )
    y.append(sentence[4])
#    print len(X)
#    print len(y)
print len(X)
print len(y)

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
X = v.fit_transform(X)

cutoff = int(.75 * len(X))
training_sentences = X[:cutoff]
training_tags = y[:cutoff]
test_sentences = X[cutoff:]
test_tags = y[cutoff:]
 
print len(training_sentences)   
print len(test_sentences)


# clf = Pipeline([
#     ('vectorizer', DictVectorizer(sparse=False)),
#     # ('classifier', DecisionTreeClassifier(criterion='entropy'))
#     ('classifier', SGDClassifier(loss='log'))
# #    ('classifier', svm.SVC(gamma=0.001, C=100.))
# ])
# v = DictVectorizer(sparse=False)
# print training_sentences_1
# training_sentences = v.fit_transform(training_sentences_1)
# training_tags = v.fit_transform(training_tags_1)

clf = SGDClassifier(loss='log')

n_iter = 15
for n in range(n_iter):
    clf.partial_fit(training_sentences[(n*10000):((n+1)*10000)], training_tags[(n*10000):((n+1)*10000)],classes=numpy.unique(training_tags))  

print 'Training completed'
 
# X_test, y_test = transform_to_dataset(test_sentences)
 
print "Accuracy:", clf.score(test_sentences, test_tags)


# print X


# print v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]

# print v.transform({'foo': 4, 'unseen_feature': 3})

