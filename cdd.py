#esse codigo eh para avaliar a qtd de grupos existem e agrupar os distribuidores

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_name(names, lat):
    for key in names['latitude']:
        if names['latitude'][key] == lat:
            return names['client_name'][key]


def elbow_curve():
    K_clusters = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = df[['latitude']]
    X_axis = df[['longitude']]
    score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
    # Visualize
    plt.plot(K_clusters, score)
    plt.xlabel('Numero de Grupos')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()


def cluster(df):
    names = df[['client_name', 'latitude']].to_dict()

    df = df.drop(['client_name', 'date'], axis=1)

    kmeans = KMeans(n_clusters=5, init='k-means++')
    kmeans.fit(df[df.columns[0:6]])
    df['cluster_label'] = kmeans.fit_predict(df[df.columns[0:6]])
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(df[df.columns[0:6]])

    # print centers

    # print labels

    length = len(df)

    df.plot.scatter(x='latitude', y='longitude', c=labels, s=100, cmap='viridis')
    center_x = []
    center_y = []
    for i in centers:
        center_x.append(i[4])
    for i in centers:
        center_y.append(i[5])
    # print(center_x)
    # print(center_y)
    plt.scatter(center_x, center_y, c='black', s=200, alpha=0.5)
    # plt.scatter(centers[5:6, 0], centers[5:6, 1], c='black', s=200, alpha=0.5)
    for i in range(0, length):
        plt.annotate(get_name(names, df['latitude'][i]), (df['latitude'][i], df['longitude'][i]),
                     horizontalalignment='right', fontsize=13, verticalalignment='bottom')
    plt.title("Grupos de Bares Moema -SP")

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('dist.csv')
    elbow_curve()
    cluster(df)
