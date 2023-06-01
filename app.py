from flask import Flask, render_template, request
from scriptKNN import knn_predict_accident

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    accident_info = {
        'descr_cat_veh': int(request.form['descr_cat_veh']),
        'descr_agglo': int(request.form['descr_agglo']),
        'descr_lum': int(request.form['descr_lum']),
        'descr_athmo': int(request.form['descr_athmo']),
        'descr_etat_surf': int(request.form['descr_etat_surf']),
        'descr_type_col': int(request.form['descr_type_col']),
        'latitude': float(request.form['latitude']),
        'longitude': float(request.form['longitude'])
    }

    accidents_file = 'chemin/vers/le/fichier.csv'  # Remplacez par le chemin réel du fichier CSV

    predicted_class_json = knn_predict_accident(accident_info, accidents_file)

    return predicted_class_json

if __name__ == '__main__':
    app.run()
