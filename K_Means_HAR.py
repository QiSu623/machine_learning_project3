import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from IPython.display import display
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 4000)

data = pd.read_csv('E:/Pythontest/simplifiedhuarus/train.csv')
#print(data.sample(5))


#save label as string
Labels = data['activity']
data = data.drop(['rn','activity'], axis=1)
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)


#normalize the dataset
data = (data - np.mean(data)) / np.std(data)
#print(data)

#find the best k
km = range (1,20)
inertias = []
for k in km:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)
plt.figure(figsize=(8,5))
plt.plot(km, inertias)
plt.xticks(km)


def k_means(n_clust, data_frame, true_labels):
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust),
             silhouette_score(data_frame, y_clust, metric='euclidean')))
    centers = k_means.cluster_centers_
    print(centers)
    colors = ['r', 'c']
    plt.figure()
    for j in range(2):
        index_set = np.where(y_clust == j)
        cluster = data.iloc[index_set]
        plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], c=colors[j], marker='.')
        plt.plot(centers[j][0], centers[j][1], 'o', markerfacecolor=colors[j], markeredgecolor='k', markersize=8)
    plt.show()

#change labels into binary: 0 - not moving, 1 - moving
Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1
Labels_binary = np.array(Labels_binary.astype(int))

k_means(n_clust=2, data_frame=data,true_labels=Labels_binary)