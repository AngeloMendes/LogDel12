
# esse codigo eh o modelo de predicao para indicar ao cliente qual melhor distribuidor

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

# leitura dos pedidos para treinar e testar o modelo
my_data = pd.read_csv('orders_predict.csv')
X = np.asarray(my_data[['qtd_antartica', 'qtd_brahma', 'qtd_skol', 'qtd_corona', 'latitude', 'longitude']])
y = np.asarray(my_data[['class']]).ravel()
best_dist = my_data[['client_name', 'latitude', 'longitude', 'class']]

score = -1
best_result = 0

X_train = np.array(1)
X_test = np.array(1)
y_train = np.array(1)
y_test = np.array(1)

clf = None
for i in range(0, 19):

    seed = random.randint(1, 101)
    if score < best_result:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    # print(score)
    if best_result < score:
        best_result = score

# predicao do melhor distribuidor de acordo com perfil e distancia
aux = 0
dist_recomend = {'distribuidor': '', 'distancia': 0}
for test in X_test:
    predict = clf.predict([test])
    # print("predicao: "+str(predict))
    # print("classe: "+str(y_test[aux]))
    aux += 1
    best_dist = np.asarray(best_dist.loc[best_dist['class'].isin([predict[0]])])
    for bar in best_dist:
        dist = np.linalg.norm(test[4:6] - bar[1:3])
        if dist_recomend['distancia'] > dist or dist_recomend['distancia'] == 0:
            dist_recomend['distancia'] = round(dist, 2)
            dist_recomend['distribuidor'] = bar[0]
    best_dist = my_data[['client_name', 'latitude', 'longitude', 'class']]

# a saida eh: {'distribuidor': 'Camel Bar', 'distancia': 5440.9}
print(dist_recomend)
