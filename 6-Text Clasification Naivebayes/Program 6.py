from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
"""
subset : ‘train’ or ‘test’, ‘all’
    Select the dataset to load: ‘train’ for the training set, ‘test’ for the test set, ‘all’ for both, with shuffled ordering.
categories : None or collection of string or unicode
    If None (default), load all the categories. If not None, list of category names to load (other categories ignored).
shuffle : bool, optional
    Whether or not to shuffle the data: might be important for models that make the assumption that the samples are independent
    and identically distributed (i.i.d.), such as stochastic gradient descent.

"""
twenty_train =fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)

""" printing one email thread """
print("\n ".join(twenty_train.data[0].split("\n")))
print("------------------------------------------")
print("tt",twenty_train.target[0])
print("--------------------------------")


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
#print("count_vect",count_vect)
#print("---------------------------------")

"""
fit_transform(raw_documents)
    Where raw_documents : iterable
        An iterable which yields either str, unicode or file objects.
Learn the vocabulary dictionary and return term-document matrix.
REturns arrray viz. document term matrix
"""
X_train_tf = count_vect.fit_transform(twenty_train.data)
#print("X_train_tf",X_train_tf)

from sklearn.feature_extraction.text import TfidfTransformer
"""
Transform a count matrix to a normalized tf [term-frequency] or tf-idf [term-frequency times inverse document-frequency]
representation.
The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down
the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than
features that occur in a small fraction of the training corpus.
"""
tfidf_transformer = TfidfTransformer()
print("-------------------------------")
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
#print("X_train_tfidf",X_train_tfidf)
X_train_tfidf.shape

"""
MultinomialNB - Naive Bayes classifier for multinomial models.


"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
mod = MultinomialNB()

mod.fit(X_train_tfidf, twenty_train.target)

X_test_tf = count_vect.transform(twenty_test.data)

X_test_tfidf = tfidf_transformer.transform(X_test_tf)
predicted = mod.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(twenty_test.target, predicted))
print("accu--------------------------------------------")
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
print("class------------------------------------------------------------")

print("confusion matrix is \n",metrics.confusion_matrix(twenty_test.target, predicted))
