import pandas as pd
from sklearn.svm import SVR
import imp
from sklearn.neural_network import MLPRegressor
from sklearn import metrics, datasets
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def list_float(start,N):
  lista = []
  counter = start
  while counter <= N:
      lista.append(counter)
      counter += 0.2
  return lista

db = pd.read_csv('Base_teste.csv')

X = db[db.columns[2]]
Y = db[db.columns[8]]

n_samples = len(X)
divisao = 0.70

order = np.random.permutation(n_samples)

X = X[order]
Y = Y.sort_values()

X_teste = X[int(divisao*n_samples):]
Y_teste = Y[int(divisao*n_samples):]

X_treino = X[:int(divisao*n_samples)]
Y_treino = Y[:int(divisao*n_samples)]


plt.rcParams['figure.figsize'] = (11,7)

plt.ylabel("Precos", color='#22a6b3', size = 19)
plt.xlabel("Marcas", color ='#22a6b3', size = 19)
plt.title("Valores de celulares a partir do preço", color = '#22a6b3', size = 20)
plt.tick_params(axis='y',color ='#f9ca24', size = 20)
plt.tick_params(axis='x',color ='#f9ca24', size = 20)
plt.grid(axis='y', linestyle='-', color = '#dfe6e9')
plt.scatter(X_teste,Y_teste, label = 'Celulares', color = '#ffbe76')
plt.legend()

plt.show()
