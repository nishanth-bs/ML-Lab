from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
training_data =fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
testing_data = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)

""" printing one email thread """
print("\n ".join(training_data.data[0].split("\n")))
print("tt",training_data.target[0])

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
countVectorizer = CountVectorizer()
X_train_tf = countVectorizer.fit_transform(training_data.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics

multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_tfidf, training_data.target)

X_test_tf = countVectorizer.transform(testing_data.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
predicted = multinomial_nb.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(testing_data.target, predicted))
print(classification_report(testing_data.target,predicted, target_names=testing_data.target_names))

print("confusion matrix is \n",metrics.confusion_matrix(testing_data.target, predicted))
