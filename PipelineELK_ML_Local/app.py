from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return "Bienvenue dans Flask pour predir l'Intrusion!"


# Charger le modèle ML
model = joblib.load(r"D:\pfe_iset\IntegrationML_ELK\ml_pipeline_local\ml_api\xgb_model.joblib")

# Mappings pour les variables catégoriques
protocol_type_mapping = {"tcp": 0, "udp": 1, "icmp": 2}
service_mapping = {"http": 0, "ftp": 1, "smtp": 2}  # Exemple, à adapter selon votre modèle
flag_mapping = {"SF": 0, "REJ": 1, "RSTO": 2}  # Exemple, à adapter selon votre modèle

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        print("Received data:", data)  # Debugging

        # Valider les champs requis
        required_fields = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "wrong_fragment", "hot", "logged_in", "num_compromised", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encoder les variables catégoriques
        features = [
            data["duration"],
            protocol_type_mapping.get(data["protocol_type"], -1),  # -1 si non trouvé
            service_mapping.get(data["service"], -1),  # -1 si non trouvé
            flag_mapping.get(data["flag"], -1),  # -1 si non trouvé
            data["src_bytes"], data["dst_bytes"], data["wrong_fragment"], data["hot"],
            data["logged_in"], data["num_compromised"], data["count"], data["srv_count"],
            data["serror_rate"], data["srv_serror_rate"], data["rerror_rate"]
        ]

        # Vérifier si des encodages ont échoué
        if -1 in features:
            return jsonify({"error": "Invalid categorical value in input data"}), 400

        print("Features:", features)  # Debugging

        # Faire une prédiction
        prediction = model.predict([features])[0]
        return jsonify({"prediction": int(prediction)})  # Convertir en type natif Python

    except Exception as e:
        # Gérer les erreurs inattendues
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)




# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # ← autorise toutes les origines


# from flask import Flask, jsonify
# import subprocess, json

# app = Flask(__name__)

# @app.route('/start')
# def start():
#     subprocess.Popen(["start", "run_pipelinev2.bat"], shell=True)
#     return "Démarrage en cours."

# @app.route('/stop')
# def stop():
#     subprocess.Popen(["start", "stop_pipeline.bat"], shell=True)
#     return "Arrêt en cours."

# @app.route('/logs')
# def logs():
#     # Exemple de récupération de logs depuis un fichier ou Elastic
#     try:
#         with open("D:/pfe_iset/IntegrationML_ELK/logs_recent.json", "r") as f:
#             data = json.load(f)
#         return jsonify(data)
#     except:
#         return jsonify({"error": "Logs non disponibles."})
