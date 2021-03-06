import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

from matplotlib import pyplot
colormap = np.array(['red', 'lime', 'black'])
def plot(startAt,x,y,c,title):
        pyplot.subplot(1,2,startAt)
        pyplot.scatter(x,y, c=c, s=40)
        pyplot.title(title)

pyplot.figure(figsize=(14,7))
plot(1, X.Sepal_Length, X.Sepal_Width, colormap[y.Targets], 'Sepal')
plot(2, X.Petal_Length, X.Petal_Width, colormap[y.Targets], 'Petal')
pyplot.show()

l1 = [0,1,2]
def rename(s):
	l2 = []
	for i in s:
		if i not in l2:
			l2.append(i)

	for i in range(len(s)):
		pos = l2.index(s[i])
		s[i] = l1[pos]
		#print("values",pos,s[i])
	return s

import sklearn.metrics as metrics
def conclude(labels,title):
        renamed = []
        renamed = rename(labels)
        print("Actual Target is:\n", iris.target)
        print("\nWhat ",title," thought: \n", renamed)
        print("Accuracy of ",title," is ",metrics.accuracy_score(y, renamed))
        print("Confusion Matrix for ",title," is \n",metrics.confusion_matrix(y, renamed))
        
#--------------KMeans Clustering----------------
from sklearn.cluster import KMeans

kMeansModel = KMeans(n_clusters=3).fit(X)
pyplot.figure(figsize=(14,7))
plot(1,X.Petal_Length,X.Petal_Width, colormap[y.Targets], 'Real Classification')
plot(2,X.Petal_Length,X.Petal_Width, colormap[kMeansModel.labels_], 'K Means Classification')
pyplot.show()
conclude(kMeansModel.labels_,'Kmeans')

#-----------------------------------------------
#---------EM ALgorithm--------------------------
from sklearn.preprocessing import StandardScaler
preprocessed = StandardScaler().fit(X).transform(X)
preprocessedDF = pd.DataFrame(preprocessed, columns = X.columns)
print("\nsample",preprocessedDF.sample(5))

from sklearn.mixture import GaussianMixture
gmModel= GaussianMixture(n_components=3).fit(preprocessedDF).predict(preprocessedDF)
pyplot.figure(figsize=(14,7))
plot(1,X.Petal_Length,X.Petal_Width, colormap[y.Targets], 'Real Classification')
plot(2,X.Petal_Length,X.Petal_Width, colormap[gmModel], 'GMM Classification')
pyplot.show()
conclude(gmModel,'EM')
