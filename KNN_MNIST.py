import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from numpy import linspace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def evaluate_classifier(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(trainX, trainY)
    return clf.score(testX,testY)

train = pd.read_csv('E:/Pythontest/train.csv')
test = pd.read_csv('E:/Pythontest/test.csv')
target = train["label"]
train = train.drop("label",1)

plt.figure(figsize(5, 5))
for digit_num in range(0,64):
    plt.subplot(8, 8, digit_num + 1)
    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    plt.xticks([])
    plt.yticks([])
#plt.show()


n_estimators_array = np.array([1,5,10,50,100,200,500])
n_samples = 10
n_grid = len(n_estimators_array)
score_array_mu =np.zeros(n_grid)
score_array_sigma = np.zeros(n_grid)
j=0
for n_estimators in n_estimators_array:
    score_array=np.zeros(n_samples)
    for i in range(0,n_samples):
        clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")
        score_array[i] = evaluate_classifier(clf, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)
    j=j+1
plt.figure(figsize(7, 3))
plt.errorbar(n_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
plt.xscale("log")
plt.xlabel("number of estimators", size = 20)
plt.ylabel("accuracy", size = 20)
plt.xlim(0.9, 600)
plt.grid(which="both")
#plt.show()

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(0,10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances

plt.figure(figsize(7, 3))
plt.plot(indices[:], importances[indices[:]], 'k.')
plt.yscale("log")
plt.xlabel("feature", size=20)
plt.ylabel("importance", size=20)



pca = PCA(n_components=2)
pca.fit(train)
transform = pca.transform(train)

plt.figure(figsize(6, 5))
plt.scatter(transform[:,0],transform[:,1], s=20, c = target, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
plt.clim(0, 9)

plt.xlabel("PC1")
plt.ylabel("PC2")


clf = KNeighborsClassifier()
n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
score_array = np.zeros(len(n_components_array))
i=0

for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    transform = pca.transform(train.iloc[0:1000])
    score_array[i] = evaluate_classifier(clf, transform, target.iloc[0:1000], 0.8)
    i=i+1
plt.figure(figsize(8,4))
plt.plot(n_components_array,score_array,'k.-')
plt.xscale('log')
plt.xlabel("number of PCA components", size=20)
plt.ylabel("accuracy", size=20)
plt.grid(which="both")
plt.show()


''''
# PCA + kNN

pca = PCA(n_components=50)
pca.fit(train)
transform_train = pca.transform(train)
transform_test = pca.transform(test)

clf = KNeighborsClassifier()
clf.fit(transform_train, target)
results=clf.predict(transform_test)

# prepare submit file

np.savetxt('knn_PCA_results.csv',
           np.c_[range(1,len(test)+1),results],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')
'''
# random forest classification
clf = RandomForestClassifier(n_estimators = 100, n_jobs=1, criterion="gini")
clf.fit(train, target)
results=clf.predict(test)

# prepare submit file

np.savetxt('Random_results.csv',
           np.c_[range(1,len(test)+1),results],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')

