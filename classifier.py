from preprocessing import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())
import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(8,6))
fig = plt.figure()
hist = train_df['label'].hist()
plt.show()

train_feature_vectors = np.load("train_feature_vectors.npy")
test_feature_vectors = np.load("test_feature_vectors.npy")
#print(train_feature_vectors.shape)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
#svclassifier = SVC(kernel='linear')
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
#text_clf = Pipeline([('clf', MultinomialNB())])
#text_clf.fit(train_df['review'], train_df['label'])
text_clf.fit(train['review'], train_df['label'])

#predicted = text_clf.predict(test_df['review'])
predicted = text_clf.predict(test['review'])
print(np.mean(predicted == test_df['label']))

from sklearn.metrics import precision_recall_fscore_support

metrics = precision_recall_fscore_support( test_df['label'], predicted)
print(precision_recall_fscore_support( test_df['label'], predicted))
print("Test precision Recall F1score: ", metrics[0],metrics[1],metrics[2])

##########################################################################################################################################
###############################SVM was better but it was taking too long to train##########
text_clf2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SVC(kernel='linear'))])
#text_clf = Pipeline([('clf', MultinomialNB())])
#text_clf.fit(train_df['review'], train_df['label'])
text_clf2.fit(train['review'], train_df['label'])
predicted = text_clf2.predict(test['review'])
print(np.mean(predicted == test_df['label']))

metrics = precision_recall_fscore_support( test_df['label'], predicted)
print(precision_recall_fscore_support( test_df['label'], predicted))
print("Test precision Recall F1score: ", metrics[0],metrics[1],metrics[2])
text_clf2 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SVC(kernel='linear'))])
from sklearn.model_selection import GridSearchCV
#param_grid = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01],'kernel': ['rbf', 'poly', 'sigmoid']}
param_grid = {'kernel': ['rbf', 'poly', 'sigmoid']}
grid_search = GridSearchCV(text_clf2, param_grid=param_grid, n_jobs=-1, verbose=10, scoring="neg_log_loss")

grid_search.fit(test['review'], test_df['label'])

cv_results = grid_search.cv_results_

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(params, mean_score)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))