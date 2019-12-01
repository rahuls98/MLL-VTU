#import libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#define features of interest
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

#get train and test sets
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

#instantiate the necessary functions
cv = CountVectorizer()
tt = TfidfTransformer()
mod = MultinomialNB()

#preprocess training data
train_cv = cv.fit_transform(train.data)
train_tt = tt.fit_transform(train_cv)

#perform learning
mod.fit(train_tt, train.target)

#preprocess testing data
test_cv = cv.transform(test.data)
test_tt = tt.transform(test_cv)

#get predictions 
pred = mod.predict(test_tt)

#get accuracy, precision and recall for data
acc = accuracy_score(test.target, pred)
rep = classification_report(test.target, pred, target_names=test.target_names)
print("Accuracy = ", acc)
print("\n",rep)