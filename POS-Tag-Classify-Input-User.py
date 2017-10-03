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
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from nltk.tokenize import word_tokenize
import numpy
import json
import sys

X = []
y = []
f = open('id-ud-train.conllu', 'r')
for line in f:
  sentence = line.split("\t")
  if sentence[0].isdigit():
    data = {
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
      'prev_word': '' if sentence[0] == '1' else X[len(X)-1]['word'],
      'has_hyphen': '-' in sentence[1],
      'is_numeric': sentence[1].isdigit(),
      'capitals_inside': sentence[1][1:].lower() != sentence[1][1:] 
    }

    X.append(data)
    y.append(sentence[3])

input_sentence = raw_input("Masukkan kalimant : ")
words = word_tokenize(input_sentence)
counter = 0
for x in words:
    X.append( {
      'word': x,
      'is_first': counter == 0,
      'is_last': counter == len(words)-1,
      'is_capitalized': x[0].upper() == x[0],
      'is_all_caps': x.upper() == x,
      'is_all_lower': x.lower() == x,
      'prefix-1': x[0],
      'prefix-2': x[:2],
      'prefix-3': x[:3],
      'suffix-1': x[-1],
      'suffix-2': x[-2:],
      'suffix-3': x[-3:],
      'prev_tag': '',
      'prev_word': '' if counter == 0 else X[len(X)-1]['word'],
      'is_numeric': x.isdigit() }
    )
    counter += 1
print "List kata"
print words;
v = DictVectorizer(sparse=True)
X = v.fit_transform(X)

#97531
cutoff = 97531
training_sentences = X[:cutoff]
training_tags = y[:cutoff]
inputUser = X[cutoff:]


epoch = 200
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(17, 17,17,17), random_state=1,max_iter=epoch)
clf = SGDClassifier(loss='log')

clf.fit(training_sentences, training_tags)  
print "List Tag"
result = clf.predict(inputUser)
printRes = '['
for i in result:
    printRes += "'"+i+"'"
    printRes += " "
printRes += ']'
print printRes
