# -*- coding: utf-8 -*-
"""
@author: MO
"""
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit


from subprocess import check_output

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def encode(train, test):
	le = LabelEncoder().fit(train.species) 
	labels = le.transform(train.species)           # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	test_ids = test.id                             # save test ids for submission
    
	train = train.drop(['species', 'id'], axis=1)  
	test = test.drop(['id'], axis=1)
    
	return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)

train.head(1)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
for train_index, test_index in sss:
	X_train, X_test = train.values[train_index], train.values[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV as cc

clf = ExtraTreesClassifier(n_estimators=900 )
clf = cc(clf, cv=10, method='sigmoid')
clf.fit(train, labels)

predictions = clf.predict_proba(test)
np.shape(predictions)

sub = pd.DataFrame(predictions, columns=classes)
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('submit5.csv', index = False)
sub.head()

