import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import json

def knn_predict_accident(accident_info, accidents_file):
    # Charger le fichier CSV des accidents dans une DataFrame
    accidents_data = pd.read_csv(accidents_file, delimiter=";")

    accidents_data['descr_grav'] = accidents_data['descr_grav'].replace({
        'Indemne': 'pas grave',
        'Blessé léger': 'pas grave',
        'Blessé hospitalisé': 'grave',
        'Tué': 'grave'
    })
    columns_to_keep = ['descr_grav', 'latitude', 'longitude', 'descr_cat_veh',
                       'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf',
                       'descr_type_col']

    accidents_data = accidents_data[columns_to_keep]

    # Liste des colonnes à convertir
    columns_to_convert = ['descr_grav', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum',
                          'descr_etat_surf', 'descr_type_col']

    #convertion en float des latitudes et des longitudes
    accidents_data["latitude"] = accidents_data["latitude"].str.replace(',', '.')
    accidents_data["latitude"] = accidents_data["latitude"].astype(float)
    accidents_data["longitude"] = accidents_data["longitude"].str.replace(',', '.')
    accidents_data["longitude"] = accidents_data["longitude"].astype(float)
  
    #affichage de l'accident avant encoding
    #print("Accidents graves avant le label encoding :")
    #accident_228 = accidents_data.iloc[228]
    #print(accident_228)

    
    # Création de l'objet LabelEncoder
    le = LabelEncoder()
    for var in columns_to_convert:
        accidents_data[var] = le.fit_transform(accidents_data[var])
  
    #affichage d'un accident garve ici le 228
    #print("Accident numéro 229 après le label encoding :")
    #accident_228 = accidents_data.iloc[228]  # Accidents_data est indexé à partir de zéro, donc 227 correspond à l'indice 228
    #print(accident_228)

    # Séparer les features (X) et les labels (y) à partir du jeu de données après le label encoding
    X = accidents_data.drop('descr_grav', axis=1)
    y = accidents_data['descr_grav']

    # Créer un modèle KNN avec k=5
    knn = KNeighborsClassifier(n_neighbors=5)

    # Ajuster le modèle aux données d'entraînement
    knn.fit(X, y)

    # Réorganiser les colonnes dans le jeu de données de test
    #accident_info_encoded = {}

    accident_info_reordered = pd.DataFrame.from_dict([accident_info], orient='columns')
    accident_info_reordered = accident_info_reordered[X.columns]

    # Prédire la classe de l'accident donné
    predicted_class = knn.predict(accident_info_reordered)

    #affichage de la réponse
    #print('predicted_classe')
    #print(predicted_class)
   

    # Convertir la prédiction en "grave" ou "pas grave"
    predicted_class_string = "grave" if predicted_class[0] == 0 else "pas grave"

    # Retourner la classe de l'accident sous forme de JSON
    result = {'descr_grav': predicted_class_string}
    json_result = json.dumps(result)

    return json_result

if __name__ == '__main__':
    accident_info = {
        'descr_cat_veh': 21,
        'descr_agglo': 1,
        'descr_lum': 4,
        'descr_athmo': 0,
        'descr_etat_surf': 8,
        'descr_type_col': 1,
        'latitude': 47.1167,
        'longitude': -2.1000
    }

    accidents_file = 'export_IA2.csv'  # Remplacez par le chemin réel du fichier CSV

    predicted_class_json = knn_predict_accident(accident_info, accidents_file)
    print(predicted_class_json)
