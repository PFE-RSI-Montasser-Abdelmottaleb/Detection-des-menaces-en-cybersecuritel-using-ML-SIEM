import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# 📥 Charger le dataset NSL-KDD (assurez-vous que le chemin est bon)
df = pd.read_csv("KDDTest+.txt", header=None)

# 🧩 Définir les colonnes (selon la documentation NSL-KDD)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


df.columns = columns

# 🎯 Garder uniquement les colonnes utilisées dans simulate_logs.py
features_to_keep = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "wrong_fragment", "hot", "logged_in", "num_compromised", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate"
]
df = df[features_to_keep + ["label"]]

# 🎯 Binaire : attaque (1) ou normale (0)
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# 🧠 Encoder les catégories comme dans Flask
protocol_mapping = {"tcp": 0, "udp": 1, "icmp": 2}
service_mapping = {"http": 0, "ftp": 1, "smtp": 2}
flag_mapping = {"SF": 0, "REJ": 1, "RSTO": 2}

df = df[
    df["protocol_type"].isin(protocol_mapping)
    & df["service"].isin(service_mapping)
    & df["flag"].isin(flag_mapping)
]

df["protocol_type"] = df["protocol_type"].map(protocol_mapping)
df["service"] = df["service"].map(service_mapping)
df["flag"] = df["flag"].map(flag_mapping)

# 📊 X / y
X = df[features_to_keep]
y = df["label"]

# ⚖️ Calcul automatique du déséquilibre
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# 🔀 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Modèle XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 💾 Sauvegarder le modèle
joblib.dump(model, "xgb_model1.joblib")
print("✅ Modèle entraîné et sauvegardé sous xgb_model1.joblib")

# 📈 Facultatif : évaluer
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
