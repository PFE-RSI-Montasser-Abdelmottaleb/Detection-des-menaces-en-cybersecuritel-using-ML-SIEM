# Importation des librairies nécessaires pour l'analyse, la visualisation et le traitement des données
import numpy as np           # Calcul scientifique
import pandas as pd          # Manipulation de données
import matplotlib.pyplot as plt  # Tracés graphiques
import seaborn as sns        # Visualisation avancée
import warnings              # Gestion des avertissements

# Ignorer tous les warnings pour une sortie plus propre
warnings.filterwarnings('ignore')  # Ignore les warnings

# Configurer pandas pour afficher les floats avec 3 chiffres après la virgule
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Affiche les floats avec 3 décimales

# Définir la taille par défaut des figures matplotlib
plt.rcParams["figure.figsize"] = (5,3)  # Définit la taille des figures

# Charger le dataset depuis un fichier CSV
df = pd.read_csv("nsl_kdd.csv")         # Charge le dataset
print("Initial DataFrame shape :", df.shape)  # Affiche la taille du DataFrame
df.info()                               # Infos sur les colonnes et types
df.describe().T                         # Statistiques descriptives

# Vérifier la présence de valeurs nulles par colonne
print("Valeurs nulles par colonne :")   # Vérifie les valeurs nulles
print(df.isnull().sum())                # Affiche le nombre de valeurs nulles

# Fonction pour analyser rapidement les valeurs uniques dans une ou plusieurs colonnes
def unique_values(df, columns):         # Fonction pour analyser les valeurs uniques
    for column_name in columns:
        print(f"\nAnalyse de la colonne : {column_name}")  # Affiche le nom de la colonne
        unique_vals = df[column_name].unique()             # Valeurs uniques
        value_counts = df[column_name].value_counts()      # Compte des valeurs
        print(f"Nombre de valeurs uniques : {len(unique_vals)}")  # Nombre de valeurs uniques
        print(f"Valeurs uniques : {unique_vals}")          # Affiche les valeurs uniques
        print(f"Comptes des valeurs : \n{value_counts}\n") # Affiche le compte
        print('='*40)                                      # Séparateur

# Sélectionner les colonnes de type 'object' (c'est-à-dire catégorielles)
cat_features = df.select_dtypes(include='object').columns  # Colonnes catégorielles
# Analyser les valeurs uniques dans ces colonnes
unique_values(df, cat_features)                            # Analyse des valeurs uniques

# Vérifier la présence de doublons dans le dataset
doublons = df.duplicated().sum()           # Nombre de doublons
print(f"Nombre de doublons : {doublons}")  # Affiche le nombre de doublons

# Transformation de la colonne 'attack' : si 'normal', sinon 'attack'
print("Transformation de la colonne 'attack' en catégories...")  # Message
attack_n = ['normal' if i == 'normal' else "attack" for i in df['attack']]  # Binarise 'attack'
df['attack'] = attack_n                   # Remplace la colonne
# Vérifier que la transformation s'est bien passée
print("Exemple de nouvelles valeurs dans 'attack' :", df['attack'].unique())  # Vérifie

# Visualiser la distribution des attaques en fonction du protocole
print("Distribution des attaques selon le protocole...")  # Message
plt.figure(figsize=(16,4))                # Taille du graphique
sns.countplot(x='attack', data=df, hue='protocol_type')  # Compte attaques/protocole
plt.xticks(rotation=45)                   # Rotation des labels
plt.title('Nombre d\'attaques par type de protocole')     # Titre
plt.show()                                # Affiche

# Vérifier la proportion de chaque type de protocole dans le dataset
protocole_counts = df["protocol_type"].value_counts(normalize=True)  # Proportion protocoles
print("Proportion des protocoles :")      # Message
print(protocole_counts)                   # Affiche les proportions

# Visualiser la répartition selon la variable 'is_guest_login' avec attaque en hue
print("Répartition en fonction de 'is_guest_login'...")   # Message
plt.figure(figsize=(10, 6))              # Taille du graphique
sns.countplot(x='is_guest_login', hue='attack', data=df, palette='Set2')  # Compte invité/attaque
plt.xlabel('Connexion invité')           # Label X
plt.ylabel('Nombre')                     # Label Y
plt.title('Répartition des attaques selon invité')        # Titre
plt.legend(title='Type d\'attaque')      # Légende
plt.grid(True)                           # Grille
plt.show()                               # Affiche

# Encodage des variables catégorielles en numériques via LabelEncoder
print("Encodage des variables...")       # Message
from sklearn import preprocessing        # Import preprocessing
le = preprocessing.LabelEncoder()        # Crée un encodeur
clm = ['protocol_type', 'service', 'flag', 'attack']  # Colonnes à encoder
# Appliquer l'encodage à chaque colonne catégorielle
for col in clm:                          # Pour chaque colonne
    df[col] = le.fit_transform(df[col])  # Encode la colonne
    print(f"Encoding de la colonne {col} terminé.")  # Message

# Séparer le dataset en features (X) et cible (y)
from sklearn.model_selection import train_test_split  # Import split
X = df.drop(["attack"], axis=1)          # Variables explicatives
y = df["attack"]                         # Variable cible
# Diviser en jeu d'entraînement et de test (90% / 10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)  # Split
print("Shapes après séparation :")       # Message
print(f"X_train : {X_train.shape}, X_test : {X_test.shape}")  # Affiche tailles

# Sélectionner les variables importantes via l'information mutuelle
from sklearn.feature_selection import mutual_info_classif  # Import MI
print("Calcul des scores d'information mutuelle...")       # Message
mutual_info_scores = mutual_info_classif(X_train, y_train) # Calcule MI
# Créer une série pour visualiser
mutual_info_series = pd.Series(mutual_info_scores, index=X_train.columns)  # Série MI
# Trier par ordre décroissant
mutual_info_series.sort_values(ascending=False, inplace=True)  # Trie
print("Top 10 variables selon l'information mutuelle :")   # Message
print(mutual_info_series.head(10))                         # Affiche top 10

# Visualisation de l'importance des variables
print("Visualisation des variables importantes...")        # Message
mutual_info_series.plot.bar(figsize=(20,5))                # Barplot MI
plt.title("Importance des variables via Mutuelle d'Information")  # Titre
plt.ylabel('Score')                                        # Label Y
plt.show()                                                 # Affiche

# Sélectionner les 15 meilleures variables selon l'information mutuelle
from sklearn.feature_selection import SelectKBest           # Import SelectKBest
selector = SelectKBest(mutual_info_classif, k=15)          # Sélecteur 15 meilleures
selector.fit(X_train, y_train)                             # Fit
# Récupérer les colonnes sélectionnées
selected_cols = X_train.columns[selector.get_support()]    # Colonnes sélectionnées
print("Variables sélectionnées :", selected_cols.tolist()) # Affiche sélection

# Visualiser l'importance de ces 15 variables
plt.figure(figsize=(12,8))                                 # Taille graphique
sns.barplot(y=selected_cols, x=mutual_info_series[selected_cols])  # Barplot sélection
plt.xlabel('Score d\'importance')                          # Label X
plt.title('Variables sélectionnées et leurs scores')        # Titre
plt.show()                                                 # Affiche

# Réduire les jeux d'entraînement et de test aux variables sélectionnées
X_train = X_train[selected_cols]                           # Réduit X_train
X_test = X_test[selected_cols]                             # Réduit X_test

# Normaliser les données pour les modèles
from sklearn.preprocessing import StandardScaler           # Import scaler
scaler = StandardScaler()                                 # Crée scaler
X_train_scaled = scaler.fit_transform(X_train)            # Normalise train
X_test_scaled = scaler.transform(X_test)                  # Normalise test

print("Données normalisées et prêtes pour la modélisation.")  # Message

# Importer et instancier les modèles
from sklearn.linear_model import LogisticRegression        # Import LR
from xgboost import XGBClassifier                         # Import XGBoost

# Créer les objets modèles
XGB_model = XGBClassifier(random_state=42)                # Modèle XGBoost
Logistic_model = LogisticRegression(random_state=42)      # Modèle LR

# Entraîner les modèles sur les données normalisées
XGB_model.fit(X_train_scaled, y_train)                    # Entraîne XGBoost
Logistic_model.fit(X_train_scaled, y_train)               # Entraîne LR

print("Modèles entraînés avec succès.")                   # Message

# Fonction pour évaluer chaque modèle avec diverses métriques
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score  # Import métriques
def eval_metric(model, X_train, y_train, X_test, y_test):
    print("\n--- Évaluation du modèle ---")  # Début de l'évaluation

    print("Jeu de test :")  # Indique l'évaluation sur le jeu de test
    print("Matrice de confusion :")  # Annonce la matrice de confusion
    y_test_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_test_pred))  # Affiche la matrice de confusion

    print("Rapport de classification :")  # Annonce le rapport de classification
    print(classification_report(y_test, y_test_pred))  # Affiche le rapport

    print("\nJeu d'entraînement :")  # Indique l'évaluation sur le jeu d'entraînement
    print("Matrice de confusion :")  # Annonce la matrice de confusion
    y_train_pred = model.predict(X_train)
    print(confusion_matrix(y_train, y_train_pred))  # Affiche la matrice de confusion

    print("Rapport de classification :")  # Annonce le rapport de classification
    print(classification_report(y_train, y_train_pred))  # Affiche le rapport

    print("="*50)  # Séparateur visuel

# Évaluer le modèle XGBoost avec la fonction personnalisée
eval_metric(XGB_model, X_train_scaled, y_train, X_test_scaled, y_test)  # Évalue XGB

# Rapport de classification détaillé pour XGBoost
print("\n===== Rapport de classification : XGBoost =====")
print("Ce rapport montre la précision, le rappel et le f1-score pour chaque classe (0=Normal, 1=Attack).")
print("Un score parfait (1.00) indique que le modèle prédit parfaitement toutes les classes.\n")
y_pred_xgb = XGB_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Attack']))

# Rapport de classification détaillé pour Logistic Regression
print("\n===== Rapport de classification : Logistic Regression =====")
print("Ce rapport montre la précision, le rappel et le f1-score pour chaque classe (0=Normal, 1=Attack).")
print("Plus les scores sont proches de 1, meilleur est le modèle.\n")
y_pred_log = Logistic_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_log, target_names=['Normal', 'Attack']))

# Analyse comparative
print("\n===== Analyse comparative =====")
print("Logistic Regression :")
print("- Précision, rappel et f1-score autour de 0.90 sur le jeu de test.")
print("- Bon équilibre entre les classes, mais quelques erreurs de classification subsistent.")
print("- Moins complexe, plus rapide à entraîner, facile à interpréter.\n")
print("XGBoost :")
print("- Précision, rappel et f1-score de 1.00 sur le jeu de test et d’entraînement.")
print("- Prédit parfaitement toutes les classes (aucune erreur).")
print("- Modèle plus complexe, risque de surapprentissage (overfitting) si les scores sont parfaits sur train ET test.\n")
print("Conclusion :")
print("Si tu recherches la performance maximale et que tu n’as pas de problème d’overfitting, XGBoost est le meilleur choix.")
print("Si tu veux un modèle plus simple et plus interprétable, Logistic Regression peut suffire.")
print("Dans ton cas, XGBoost est le plus performant, mais vérifie que le jeu de test est bien représentatif pour éviter un surapprentissage caché.")

from sklearn.metrics import confusion_matrix, classification_report  # Import des métriques

y_pred = XGB_model.predict(X_test_scaled)  # Prédictions sur le jeu de test normalisé
cm = confusion_matrix(y_test, y_pred)      # Calcul de la matrice de confusion

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Affiche la matrice de confusion
plt.xlabel('Prédiction')  # Axe X
plt.ylabel('Réel')        # Axe Y
plt.title('Matrice de confusion du modèle XGBoost')  # Titre
plt.show()  # Affiche le graphique

print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))  # Rapport de classification
# Optimisation des hyperparamètres pour XGBoost avec GridSearchCV
print("Recherche des meilleurs hyperparamètres pour XGBoost...")  # Message
param_grid = {                                    # Grille de paramètres
    "n_estimators": [50, 64, 100, 128],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.5, 0.8],
    "colsample_bytree": [0.5, 0.8]
}

from sklearn.model_selection import GridSearchCV   # Import GridSearch
grid_search = GridSearchCV(
    XGBClassifier(random_state=42), param_grid, scoring="f1", n_jobs=-1, return_train_score=True
)  # Crée GridSearch
grid_search.fit(X_train_scaled, y_train)           # Lance la recherche

# Afficher le meilleur score et paramètres trouvés
print(f"Meilleur score F1 : {grid_search.best_score_}")      # Affiche meilleur score
print(f"Meilleures params : {grid_search.best_params_}")     # Affiche meilleurs params

# Construire le modèle final avec les meilleurs hyperparamètres
best_params = grid_search.best_params_                       # Récupère meilleurs params
final_xgb = XGBClassifier(**best_params)                     # Nouveau modèle XGB
final_xgb.fit(X_train_scaled, y_train)                       # Entraîne modèle final

# Prédictions finales
y_pred_final = final_xgb.predict(X_test_scaled)              # Prédictions finales
# Récupérer les labels originaux pour une meilleure compréhension
y_pred_string = le.inverse_transform(y_pred_final)           # Décodage labels
# Probabilités prédites pour la classe positive
y_pred_proba = final_xgb.predict_proba(X_test_scaled)        # Probabilités

# Afficher plusieurs métriques
print("Évaluation finale du modèle optimisé :")              # Message
print("F1-score : {:.3f}".format(f1_score(y_test, y_pred_final)))  # F1-score
print("Recall : {:.3f}".format(recall_score(y_test, y_pred_final)))  # Recall
print("AUC ROC : {:.3f}".format(roc_auc_score(y_test, y_pred_proba[:,1])))  # AUC

# Courbe ROC pour visualiser la performance
from sklearn.metrics import RocCurveDisplay                  # Import ROC
RocCurveDisplay.from_estimator(final_xgb, X_test_scaled, y_test)  # Courbe ROC
plt.title('Courbe ROC du modèle final')                      # Titre
plt.show()                                                   # Affiche

# Afficher l'importance des variables selon le modèle
importances = pd.DataFrame(
    {'Variable': selected_cols, 'Importance': final_xgb.feature_importances_}
).sort_values(by='Importance', ascending=False)              # Importance variables

print("Importance des variables :")                          # Message
print(importances)                                           # Affiche importances

# Visualiser l'importance des variables
plt.figure(figsize=(12,8))                                   # Taille graphique
sns.barplot(x='Importance', y='Variable', data=importances)  # Barplot importances
plt.title('Importance des variables selon le modèle XGBoost')# Titre
plt.show()                                                   # Affiche

# Visualisation supplémentaire des variables importantes
print("Visualisation supplémentaire des variables importantes...")  # Message
plt.figure(figsize=(10,6))                                   # Taille graphique
sns.scatterplot(x=importances['Variable'], y=importances['Importance'])  # Scatterplot
plt.xticks(rotation=45)                                      # Rotation labels
plt.ylabel("Importance")                                     # Label Y
plt.title("Variables importantes selon le modèle")            # Titre
plt.show()                                                   # Affiche

# Résumé des prédictions sur le jeu de test
print("Résumé des prédictions sur le jeu de test :")         # Message
plt.figure(figsize=(10,6))                                   # Taille graphique
sns.countplot(x=y_pred_string, palette="pastel")             # Compte prédictions
plt.xlabel("Type d'attaque prédit")                          # Label X
plt.ylabel("Nombre")                                         # Label Y
plt.title("Répartition des attaques prédites")               # Titre
plt.show()                                                   # Affiche

# Sauvegarder le modèle entraîné dans un fichier
import joblib                                                # Import joblib
joblib.dump(final_xgb, 'xgb_model.joblib')                   # Sauvegarde modèle
print("Le modèle a été sauvegardé sous le nom 'xgb_model.joblib'.")  # Message

# Charger le modèle sauvegardé pour test
from joblib import load                                      # Import load
loaded_model = load('xgb_model.joblib')                      # Charge modèle
# Prédire avec le modèle chargé
predictions = loaded_model.predict(X_test_scaled)            # Prédictions
# Calculer la précision
accuracy = accuracy_score(y_test, predictions)               # Précision
print(f"Précision du modèle chargé : {accuracy:.3f}")        # Affiche précision

# Charger à nouveau le modèle pour affichage
from joblib import load                                      # Import load
model = load('xgb_model.joblib')                             # Charge modèle

# Afficher le contenu ou l’objet du modèle
print(model)                                                 # Affiche modèle

# Si c'est un XGBClassifier, afficher ses importances
importances = model.feature_importances_                     # Importances XGB
print(importances)                                           # Affiche importances