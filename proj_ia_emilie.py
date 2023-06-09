# -*- coding: utf-8 -*-
"""Proj_IA_emilie.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zZHamCeayDZ9OqRwIIF3aq2Cj8f_KUnv
"""



"""## Organisation des données"""

import pandas  as pd
data = pd.read_csv("export_IA2.csv",delimiter=";")
print(type(data))

data.info()

data.shape

data['descr_grav'].value_counts()

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

fig = px.histogram(data['descr_athmo'], x="descr_athmo", title="Nombre d'accidents en fonction des conditions athmosphérique", labels ={"descr_athmo" : "conditions athmosphérique"})
fig.show()

accidents_graves = data[data['descr_grav'] == 'Tué']
accidents_graves.head()
fig = px.histogram(accidents_graves['descr_athmo'], x="descr_athmo", title="Nombre d'accidents en fonction des conditions athmosphérique", labels ={"descr_athmo" : "conditions athmosphérique"})
fig.show()

data['descr_grav'] = data['descr_grav'].replace({
    'Indemne': 'pas grave',
    'Blessé léger': 'pas grave',
    'Blessé hospitalisé': 'grave',
    'Tué': 'grave'
})
data.head(3)

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
# Liste des colonnes à convertir
columns_to_convert = ["descr_agglo", "descr_cat_veh","descr_athmo","descr_lum","descr_etat_surf","id_code_insee","description_intersection","descr_dispo_secu","descr_dispo_secu","descr_grav","ville","descr_motif_traj","descr_type_col"]

# Création de l'objet OrdinalEncoder
le = LabelEncoder()
for var in columns_to_convert:
      data[var] = le.fit_transform(data[var])

#for column in columns_to_convert:
    # Sélection de la colonne à convertir
    #column_data = data[[column]]

    # Conversion des valeurs non numériques en utilisant OrdinalEncoder
 #   encoded_data = enc.fit_transform(column_data)

    # Conversion des valeurs encodées en entiers
  #  encoded_data = encoded_data.astype(int)

    # Remplacement de la colonne originale par les valeurs encodées entières
  #  data[[column]] = encoded_data

def convert_column_type(data, column, new_type):
    """
    Convertit le type d'une colonne spécifiée dans un dataframe sans modifier les valeurs.

    Arguments :
    - data : le dataframe contenant les données
    - column : le nom de la colonne à convertir
    - new_type : le nouveau type de données pour la colonne

    Retour :
    - Le dataframe avec le type de colonne converti
    """

    # Remplacement des virgules par des points dans la colonne
    data[column] = data[column].str.replace(',', '.')

    # Conversion du type de la colonne
    data[column] = data[column].astype(new_type)

    return data

data = convert_column_type(data, "latitude",float)
data = convert_column_type(data, "longitude",float)

def convert_dates(data, column):
    """
    Convertit les colonnes spécifiées contenant des valeurs de dates et heures en objets datetime.

    Arguments :
    - data : le dataframe contenant les données
    - columns : une liste ou un tableau contenant les noms des colonnes à convertir

    Retour :
    - Le dataframe avec les colonnes converties en objets datetime
    """ 
    # Conversion des valeurs de dates et heures en objets datetime
    data[column] = pd.to_datetime(data[column])

    return data

data = convert_dates(data, "date")

data.info()

data.head(3)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt

fig = px.scatter_mapbox(data,
                 lat=data.latitude,
                 lon=data.longitude,
                 color = data.descr_grav)
fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        center=dict(lon=2.554071, lat=46.603354),
        zoom=4
    ),
    title=dict(text="Carte des accidents en fonction de leur gravité", x=0.5)
)

fig.show()

"""##matrice de correlation

"""

corr_matrix = data.corr()
# Affichage de la matrice de corrélation
#print(corr_matrix)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration de la taille du graphique
plt.figure(figsize=(10, 8))

# Affichage de la matrice de corrélation avec des pourcentages
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu")

# Affichage du graphique
plt.show()

def keep_selected_columns(data, columns_to_keep):
    """
    Garde uniquement les colonnes spécifiées et supprime les autres colonnes d'un dataframe.

    Arguments :
    - data : le dataframe contenant les données
    - columns_to_keep : une liste ou un tableau contenant les noms des colonnes à conserver

    Retour :
    - Le dataframe avec les colonnes spécifiées conservées
    """

    # Suppression des colonnes qui ne sont pas dans la liste columns_to_keep
    columns_to_drop = set(data.columns) - set(columns_to_keep)
    data = data.drop(columns_to_drop, axis=1)

    return data

columns_to_keep = ['descr_grav', 'latitude', 'longitude', 'descr_cat_veh',
                   'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf',
                   'descr_type_col']

data = keep_selected_columns(data, columns_to_keep)

data.info()

corr_matrix_2 = data.corr()
# Configuration de la taille du graphique
plt.figure(figsize=(10, 8))

# Affichage de la matrice de corrélation avec des pourcentages
sns.heatmap(corr_matrix_2, annot=True, fmt=".2f", cmap="RdBu")

# Affichage du graphique
plt.show()

"""##Culstering"""

from sklearn.cluster import KMeans
kmean = KMeans(2)

data_filtered = data.drop(data[(data['latitude'] == 0) & (data['longitude'] == 0)].index)
X = data.loc[:, ['latitude', 'longitude']]
X = X.drop(X[(X['latitude'] == 0) & (X['longitude'] == 0)].index)

X.info()

y_predict = kmean.fit_predict(X)

fig = px.scatter_mapbox(X,
                 lat=X.latitude,
                 lon=X.longitude,
                 color = y_predict)
fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        center=dict(lon=2.554071, lat=46.603354),
        zoom=4
    ),
    title=dict(text="2 clusters", x=0.5)
)

fig.show()

"""# Etape 3

"""

from sklearn.model_selection import train_test_split

def holdout_split(data, target, test_size=0.2, random_state=42):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

from sklearn.model_selection import LeaveOneOut
import random

def leave_one_out_split(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    size = len(data)
    stop = random.randint(0, size)
    i=0
    for i, (train_index, test_index) in enumerate(loo.split(X)):
      if(i==stop):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break
    print("ligne isolé aléatoirement:",i)
    return X_train,X_test,y_train,y_test

X_train_h, X_test_h, y_train_h, y_test_h = holdout_split(data, 'descr_grav')

X_train_l, X_test_l, y_train_l, y_test_l = leave_one_out_split(data, 'descr_grav')

print(X_train_h.shape)

print(X_train_l.shape)

X_test_h.head()

"""## Classification avec KNN"""

import numpy as np
from scipy.spatial import distance
from collections import Counter

def euclidean_distance(x1, x2):
    #return x1 - x2
    x1 = np.array(x1,dtype=np.float64)
    x2 = np.array(x2,dtype=np.float64)
    return np.sqrt(np.sum(np.square(x1 - x2)))
    #return distance.euclidean(x1, x2)
class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Calculer les distances entre x et tous les exemples d'apprentissage
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        print("distance:",distances)
        # Trier les distances et obtenir les indices des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]
        print("indice:",k_indices)
        print("xtrain:",self.X_train[k_indices])
        # Extraire les étiquettes des k plus proches voisins
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Retourner l'étiquette majoritaire parmi les k plus proches voisins
        most_common = Counter(k_nearest_labels).most_common()
        return most_common

# Création de l'objet KNN avec k=5
knn = KNN(k=3)

# Ajustement du modèle aux données d'entraînement
knn.fit(X_train_l.values, y_train_l.values)

"""Notre KNN from scrach retourn les lignes les plus proche de notre X_train_l """

# Prédictions sur les données de test
y_pred_l = knn.predict(X_test_l.values)

# Affichage des k plus proches du premier
print("xtest:",y_pred_l[0])
#affiche la dernière colone: le type de colision
print("type de colision des k plus proche:")
print(le.inverse_transform(y_pred_l[0][0]))

"""visualisation de la ligne X_test_l pour comparer les k proche trouvé précedemment."""

X_test_l.values

###Trop long
#knn = KNN(k=5)
#knn.fit(X_train_h.values, y_train_h.values)
#y_pred_h = knn.predict(X_test_h.values)
#print(y_pred_h)

"""##KNN SKlearn

"""

def grave_ou_pas(nb):
  if nb==1:
    return 'pas grave'
  if nb==0:
    return 'grave'

"""###KNN sur les bases du holdout"""

from sklearn.neighbors import KNeighborsClassifier

# Création de l'objet KNeighborsClassifier avec k=5
knn_SK_h = KNeighborsClassifier(n_neighbors=5)

# Ajustement du modèle aux données d'entraînement
knn_SK_h.fit(X_train_h, y_train_h)

# Prédictions sur les données de test
y_pred_SK_h = knn_SK_h.predict(X_test_h)

# Affichage des prédictions
print("Prédictions:",y_pred_SK_h)

"""###Accuracy score avec les bases du holdout

"""

from sklearn.metrics import accuracy_score

# Calcul de l'accuracy
accuracy = accuracy_score(y_test_h, y_pred_SK_h)

# Affichage de l'accuracy
print("Accuracy:", accuracy)

"""##KNN sur la base leave one out et affiche l'accuracy et les prédiction"""

from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

def leave_one_out_knn(data, target, k=5):
    X = data.drop(target, axis=1)
    y = data[target]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    size = len(data)
    accuracy_scores = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    return accuracy_scores, average_accuracy

# Utilisation de la fonction leave_one_out_knn avec votre DataFrame
# Supposons que votre DataFrame s'appelle 'data' et la colonne cible 'target'
y_preds, avg_accuracy = leave_one_out_knn(data.sample(n=2000, random_state=42), 'descr_grav')

# Affichage des résultats
print("Prédictions:", y_preds)
print("Moyenne de l'accuracy score:", avg_accuracy)