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

default_configuration_raw = """
{
  "features": [ "word",
      "is_first",
      "is_last",
      "is_capitalized",
      "is_all_caps",
      "is_all_lower",
      "prefix-1",
      "prefix-2",
      "prefix-3",
      "suffix-1",
      "suffix-2",
      "suffix-3",
      "prev_tag",
      "prev_word",
      "has_hyphen",
      "is_numeric",
      "capitals_inside" 
  ],
  "algorithm": "MLP"
}

"""

configuration = json.loads(default_configuration_raw)

if( len(sys.argv) >= 2 ):
	config_file = sys.argv[1]
	with open(config_file,'r') as f:
		configuration = json.loads(f.read())



X = []
y = []
f = open('id-ud-train+dev.conllu', 'r')
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
      'prev_tag': '' if sentence[0] == '1' else y[len(X)-1],
      'prev_word': '' if sentence[0] == '1' else X[len(X)-1]['word'],
      'has_hyphen': '-' in sentence[1],
      'is_numeric': sentence[1].isdigit(),
      'capitals_inside': sentence[1][1:].lower() != sentence[1][1:] 
    }

    X.append({ k: v for k,v in data.items() if k in configuration["features"] })
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
      'has_hyphen': '-' in x,
      'is_numeric': x.isdigit(),
      'capitals_inside': x[1:].lower() != x[1:] }
    )
    print x
    counter += 1
    
print "Feature data size : "+str(len(X))
print "Label data size : "+str(len(y))
sizeAll = len(X)
v = DictVectorizer(sparse=True)
X = v.fit_transform(X)

print "Dividing dataset into training set and testing set ..."
#97531
cutoff = 97531

training_sentences = X[:cutoff]
training_tags = y[:cutoff]
test_sentences = X[cutoff:(sizeAll-counter)]
test_tags = y[cutoff:]
inputUser = X[(sizeAll-counter):]

print "Training set size : "+str(training_sentences.shape[0])  
print "Testing set size : "+str(test_sentences.shape[0])

# default SGD
clf = SGDClassifier(loss='log')
print "Algorithm", configuration["algorithm"]

if( configuration["algorithm"] == "SGD"):
	clf = SGDClassifier(loss='log')
elif( configuration["algorithm"] == "SVM"):
	clf = svm.SVC(decision_function_shape='ovo')
elif ( configuration["algorithm"] == "MLP" ):
	epoch = 200
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(17, 17,17,17), random_state=1,max_iter=epoch)

print 'Training started'
clf.fit(training_sentences, training_tags)  
print 'Training completed'
 
print "Testing started"
score = clf.score(test_sentences, test_tags)
print "F1 Score"
print f1_score(test_tags, clf.predict(test_sentences), average='weighted')
print "Accuracy:", score
print clf.predict(inputUser)
#
#
#clf.predict([[2., 2.], [-1., -2.]])
