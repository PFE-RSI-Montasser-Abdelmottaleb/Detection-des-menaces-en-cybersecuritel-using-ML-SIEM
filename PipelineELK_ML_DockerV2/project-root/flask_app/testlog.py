import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("xgb_model.joblib")

# Créer une entrée test (même que celle du simulateur)
data = {
    "duration": 10,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "REJ",
    "src_bytes": 50,
    "dst_bytes": 50,
    "wrong_fragment": 0,
    "hot": 0,
    "logged_in": 0,
    "num_compromised": 3,
    "count": 95,
    "srv_count": 95,
    "serror_rate": 1.0,
    "srv_serror_rate": 1.0,
    "rerror_rate": 0.0
}

df = pd.DataFrame([data])

# Appliquer les mêmes encodages que ceux utilisés à l'entraînement
df["protocol_type"] = df["protocol_type"].map({"tcp": 0, "udp": 1, "icmp": 2})
df["service"] = df["service"].map({"http": 0, "ftp": 1, "smtp": 2, "other": 3})
df["flag"] = df["flag"].map({"SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "other": 4})

# Vérifier que tous les champs sont présents
print(df)

# Faire la prédiction
pred = model.predict(df)
print("Prediction:", pred[0])
