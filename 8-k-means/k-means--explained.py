from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

l1 = [0,1,2]

#this method ensures the uniform coloring of points
#by changing any l2-[2,0,1] - [0,2,1]
#into [0,1,2]
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

from sklearn import datasets
# import dataset  
iris = datasets.load_iris()
"""
The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
"""
#print("\n IRIS DATA :",iris.data);
"""
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
         ...
 [4.7 3.2 1.3 0.2]]
"""
#print("\n IRIS FEATURES :\n",iris.feature_names) 
#print("\n IRIS TARGET  :\n",iris.target)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""
#print("\n IRIS TARGET NAMES:\n",iris.target_names)


# Store the inputs as a Pandas Dataframe and set the column names
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

#targets DataFrame obj
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

from matplotlib import pyplot
# Set the size of the plot
pyplot.figure(figsize=(14,7))
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
# Plot Sepal
"""
the subplot will take the index position on a grid with nrows rows and ncols columns.
index starts at 1 in the upper left corner and increases to the right.
"""
pyplot.subplot(1, 2, 1)#rows,column,index
pyplot.scatter(X.Sepal_Length,X.Sepal_Width, c=colormap[y.Targets], s=40)
pyplot.title('Sepal')

pyplot.subplot(1, 2, 2)
pyplot.scatter(X.Petal_Length,X.Petal_Width, c=colormap[y.Targets], s=40)
pyplot.title('Petal')
pyplot.show()

print("Actual Target is:\n", iris.target)

# K Means Cluster
model = KMeans(n_clusters=3).fit(X)

# Set the size of the plot
pyplot.figure(figsize=(14,7))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot the Original Classifications
pyplot.subplot(1, 2, 1)
pyplot.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
pyplot.title('Real Classification')

# Plot the Models Classifications
pyplot.subplot(1, 2, 2)
pyplot.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
pyplot.title('K Mean Classification')
pyplot.show()

km = rename(model.labels_)
print("\nWhat KMeans thought: \n", km)
print("Accuracy of KMeans is ",metrics.accuracy_score(y, km))
print("Confusion Matrix for KMeans is \n",metrics.confusion_matrix(y, km))


#-----------------------------------------------
#---------EM ALgorithm--------------------------
from sklearn import preprocessing
xsa = preprocessing.StandardScaler().fit(X).transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
print("\nsample",xs.sample(5))

from sklearn.mixture import GaussianMixture  #Gaussian mixture model probability distribution.
y_cluster_gmm  = GaussianMixture(n_components=3).fit(xs).predict(xs)


pyplot.subplot(1, 2, 1)
pyplot.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
pyplot.title('GMM Classification')
pyplot.show()

em = rename(y_cluster_gmm)
print("\nWhat EM thought: \n", em)
print("Accuracy of EM is ",metrics.accuracy_score(y, em))
print("Confusion Matrix for EM is \n", metrics.confusion_matrix(y, em))
