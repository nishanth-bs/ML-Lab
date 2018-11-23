from sklearn.datasets import fetch_20newsgroups

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
training_data =fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
testing_data = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)

""" printing one email thread """
print("\n ".join(training_data.data[0].split("\n")))
print("------------------------------------------")
print("tt",training_data.target[0])
print("--------------------------------")


from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer()

"""
fit_transform(raw_documents)
    Where raw_documents : iterable
        An iterable which yields either str, unicode or file objects.
Learn the vocabulary dictionary and return term-document matrix.
REturns arrray viz. document term matrix
"""
X_train_tf = countVectorizer.fit_transform(training_data.data)

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
X_train_tfidf.shape

"""
MultinomialNB - Naive Bayes classifier for multinomial models.


"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics

multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_tfidf, training_data.target)

X_test_tf = countVectorizer.transform(testing_data.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
predicted = multinomial_nb.predict(X_test_tfidf)

"""
accuracy_score(y_true, y_pred,normalize=True)
In multilabel classification, this function computes subset accuracy:
the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    Normalize if False, return the number of correctly classified samples. Otherwise, return
    the fraction of correctly classified samples.
"""
print("Accuracy:", accuracy_score(testing_data.target, predicted))
print("--------------------------------------------")

"""
classification_report(y_true, y_pred, target_names=None )
Build a text report showing the main classification metrics
    target_names : list of strings    Optional display names matching the labels (same order).

"""
print(classification_report(testing_data.target,predicted, target_names=testing_data.target_names))
print("------------------------------------------------------------")

print("confusion matrix is \n",metrics.confusion_matrix(testing_data.target, predicted))
