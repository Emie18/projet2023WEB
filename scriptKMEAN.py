import json
import numpy as np
from sklearn.cluster import KMeans

def kmeans_predict_accident(latitude, longitude, centroids):
    # Créer un array numpy à partir des coordonnées de l'accident
    accident_coordinates = np.array([[latitude, longitude]])

    # Créer un array numpy à partir de la liste des centroïdes
    centroids_array = np.array(centroids)

    # Créer un modèle K-means avec le nombre de clusters égal au nombre de centroïdes
    kmeans = KMeans(n_clusters=len(centroids_array), random_state=0,n_init=10)

    # Ajuster le modèle aux données des centroïdes
    kmeans.fit(centroids_array)

    # Prédire le cluster d'appartenance de l'accident
    predicted_cluster = kmeans.predict(accident_coordinates)

    # Retourner le cluster d'appartenance sous forme de JSON
    result = {'cluster': int(predicted_cluster)}
    json_result = json.dumps(result)

    return json_result

if __name__ == '__main__':
    # Exemple d'utilisation du script
    latitude = 47.1167
    longitude = -2.1000
    # Exemple de liste de centroïdes
    centroids = [[47.0, -2.0], [48.0, -3.0], [46.0, -2.5]]  

    predicted_cluster_json = kmeans_predict_accident(latitude, longitude, centroids)
    print(predicted_cluster_json)