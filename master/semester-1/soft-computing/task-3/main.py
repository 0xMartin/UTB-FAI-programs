import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors


def showSummary(df, title, cluster_names):
    c1, c2 = df[df['Cluster'] == 0], df[df['Cluster'] == 1]
    c1 = c1.drop('Cluster', axis=1)
    c2 = c2.drop('Cluster', axis=1)

    print("Clusters =", list(set(df["Cluster"])))

    n = len(df.columns) - 1
    figure, axis = plt.subplots(math.ceil(n / 3), 3, figsize=(16, 8))
    figure.suptitle(title)
    row_i, col_i = 0, 0
    avg1, avg2 = c1.mean(), c2.mean()
    for i, c_name in enumerate(c1.columns):
        axis[row_i, col_i].set_title(c_name)
        axis[row_i, col_i].bar(
            cluster_names, [avg1[i], avg2[i]], color='maroon', width=0.4)
        col_i += 1
        if col_i > 2:
            col_i = 0
            row_i += 1

    plt.show()


def kMeans(df):
    # urceni poctu klastru podle silhouette_score
    score_list = []
    K = range(2, 7)
    max = [0, 0]
    for k in K:
        print("%i/6" % k)
        kmeans = KMeans(init="random", n_clusters=k, metric='euclidean')
        kmeans.fit(df)
        s = silhouette_score(df, kmeans.labels_, metric='euclidean')
        if max[0] < s:
            max[0] = s
            max[1] = k
        score_list.append(s)
    print("Best number of clusters: %i" % max[1])
    plt.figure(figsize=(16, 8))
    plt.plot(K, score_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title('Silhouette Score')
    plt.show()

    # KMeans
    k = max[1]
    kmeans = KMeans(
        init="random",
        n_clusters=k,
    )
    kmeans.fit(df)

    # output
    df["Cluster"] = kmeans.labels_
    showSummary(df, "K-MEANS - k=%i" % k, ["C1", "C2"])


def get_data_radiant(data):
    return np.arctan2(data[:, 1].max() - data[:, 1].min(),
                      data[:, 0].max() - data[:, 0].min())


def find_elbow(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))
    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)
    # return index of elbow
    return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]


def dbscan(df):
    # urceni min_samples
    min_samples = 2 * (len(df.columns) - 1)

    # urceni eps
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(df)
    distances, indices = neighbors_fit.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.suptitle("Elbow rule - min samples: %i"%min_samples)
    plt.show()
    # -> 8488 -> z grafu

    # DBSCAN
    dbscan = DBSCAN(eps=8488, min_samples=min_samples, metric='euclidean')
    clustering = dbscan.fit(df)

    # output
    df["Cluster"] = clustering.labels_
    df = df.drop(df[df['Cluster'] == -1].index, inplace=False)
    showSummary(df, "DBSCAN", ["C1", "C2"])


if __name__ == '__main__':
    df = pd.read_csv('CC_GENERAL.csv')
    df = df.drop('CUST_ID', axis=1)
    df = df.dropna()

    """
    kMeans(df[['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES',
           'PAYMENTS', 'INSTALLMENTS_PURCHASES']])
    """
    dbscan(df[['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES',
           'PAYMENTS', 'INSTALLMENTS_PURCHASES']])
