from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle ML
model = joblib.load("xgb_model.joblib")

# Mappings pour les variables catégoriques (à adapter selon ton modèle)
protocol_type_mapping = {"tcp": 0, "udp": 1, "icmp": 2}
service_mapping = {"http": 0, "ftp": 1, "smtp": 2}
flag_mapping = {"SF": 0, "REJ": 1, "RSTO": 2}

# Liste des champs requis
required_fields = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "wrong_fragment", "hot",
    "logged_in", "num_compromised", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Vérifier tous les champs requis
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encoder et convertir les variables
        try:
            features = [
                float(data["duration"]),
                protocol_type_mapping.get(data["protocol_type"], -1),
                service_mapping.get(data["service"], -1),
                flag_mapping.get(data["flag"], -1),
                float(data["src_bytes"]),
                float(data["dst_bytes"]),
                float(data["wrong_fragment"]),
                float(data["hot"]),
                float(data["logged_in"]),
                float(data["num_compromised"]),
                float(data["count"]),
                float(data["srv_count"]),
                float(data["serror_rate"]),
                float(data["srv_serror_rate"]),
                float(data["rerror_rate"])
            ]
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid data type: {e}"}), 400

        # Vérifier si des encodages ont échoué
        if -1 in features[:4]:  # premières colonnes = catégorielles encodées
            return jsonify({"error": "Invalid categorical value in input data"}), 400

        print("Features:", features)

        # Prédiction
        prediction = model.predict([features])[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "bienvenue dans flask api prediction !"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
