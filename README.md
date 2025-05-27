# I Détection des Menaces en Cybersécurité avec Machine Learning et Dataset NSL-KDD 

## Introduction  
Ce projet s’inscrit dans le domaine critique de la cybersécurité, axé sur la détection d’intrusions réseau. Il exploite le dataset NSL-KDD, une référence améliorée issue du KDD Cup 1999, reconnue pour sa robustesse et son intérêt scientifique. L’objectif est d’évaluer des modèles de machine learning capables d’identifier efficacement les comportements malveillants dans le trafic réseau.

## Objectifs  
- Explorer et comprendre la structure et la qualité des données NSL-KDD.  
- Nettoyer, prétraiter et enrichir le dataset pour optimiser les performances des modèles.  
- Construire et comparer plusieurs modèles de machine learning, notamment XGBoost et la régression logistique, dédiés à la détection d’intrusions.  
- Identifier les facteurs discriminants entre trafic normal et attaques.  
- Proposer des recommandations pour améliorer les défenses réseau.

## Démarche Méthodologique  
Le projet suit une méthodologie rigoureuse comprenant :  
1. Importation des bibliothèques Python clés (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost).  
2. Chargement et préparation des jeux de données KDDTrain+ et KDDTest+.  
3. Nettoyage et traitement des valeurs manquantes et aberrantes.  
4. Analyse exploratoire des données (distribution, corrélations, tendances).  
5. Prétraitement (encodage, normalisation) pour rendre les données compatibles avec les algorithmes ML.  
6. Feature engineering pour extraire des caractéristiques pertinentes.  
7. Entraînement et optimisation des modèles XGBoost et régression logistique.  
8. Évaluation rigoureuse via métriques standard (accuracy, precision, recall, F1-score, AUC-ROC).  
9. Analyse de l’importance des variables pour interpréter les décisions du modèle.  
10. Synthèse des résultats et recommandations stratégiques pour renforcer la détection.

## Conclusion  
Ce travail met en lumière l’efficacité des techniques de machine learning dans la détection d’intrusions réseau, offrant une approche fiable et évolutive pour la protection des systèmes d’information face aux cybermenaces modernes.

#########################################################################################""

# II Pipeline ELK + Machine Learning – Version Locale (v1)
  # 1 Objectif
Ce projet implémente une architecture locale permettant de détecter des intrusions à partir de logs réseau, en temps réel, à l’aide :
      1/ d’un modèle de Machine Learning XGBoost intégré dans une API Flask,
      2/ d’une stack ELK (Elasticsearch, Logstash, Kibana),
      3/ et de Filebeat pour l’expédition des logs vers Logstash.
# 2 Architecture générale

 1/graph TD
 
    A[Serveur de logs simulé] -->|JSON logs| B[Filebeat]
    B --> C[Logstash]
    C -->|Appel HTTP| D[API Flask (ML)]
    D -->|Résultat prédiction| C
    C --> E[Elasticsearch]
    E --> F[Kibana]
    
 2/  Stack utilisée

Composant	Version	Rôle

Python	3.x	Exécution du modèle ML et scripts
Flask	2.3.3	API REST de prédiction ML
XGBoost	3.0.0	Modèle de détection d'intrusions
scikit-learn	1.3.2	Prétraitement et compatibilité ML
Filebeat	8.8.0	Expédition de logs vers Logstash
Logstash	8.8.0	Pipeline de traitement des logs
Elasticsearch	8.8.0	Indexation et stockage des événements
Kibana	8.8.0	Visualisation en temps réel

3/ Fonctionnement automatique

Le script start_pipeline.bat exécute automatiquement les composants suivants dans l’ordre :

 1 . Nettoyage de l’état Filebeat (registry)
 
 2 .Démarrage de :
 
      * Elasticsearch
      
      * Kibana
      
      * API Flask (serveur ML)
      
      * Logstash
      
      * Filebeat
      
      * Simulateur de logs JSON
      
3 . Ouverture automatique de Kibana dans le navigateur

##########################################################################################################################################""
# II  Pipeline ELK + Machine Learning – Version Locale (v1)
  # Objectif
Ce projet implémente une architecture locale permettant de détecter des intrusions à partir de logs réseau, en temps réel, à l’aide :

d’un modèle de Machine Learning XGBoost intégré dans une API Flask,

d’une stack ELK (Elasticsearch, Logstash, Kibana),

et de Filebeat pour l’expédition des logs vers Logstash.

🧩 Architecture générale
mermaid
Copier
Modifier
graph TD
    A[Serveur de logs simulé] -->|JSON logs| B[Filebeat]
    B --> C[Logstash]
    C -->|Appel HTTP| D[API Flask (ML)]
    D -->|Résultat prédiction| C
    C --> E[Elasticsearch]
    E --> F[Kibana]
⚙️ Stack utilisée
Composant	Version	Rôle
Python	3.x	Exécution du modèle ML et scripts
Flask	2.3.3	API REST de prédiction ML
XGBoost	3.0.0	Modèle de détection d'intrusions
scikit-learn	1.3.2	Prétraitement et compatibilité ML
Filebeat	8.8.0	Expédition de logs vers Logstash
Logstash	8.8.0	Pipeline de traitement des logs
Elasticsearch	8.8.0	Indexation et stockage des événements
Kibana	8.8.0	Visualisation en temps réel

🔁 Fonctionnement automatique
Le script start_pipeline.bat exécute automatiquement les composants suivants dans l’ordre :

Nettoyage de l’état Filebeat (registry)

Démarrage de :

Elasticsearch

Kibana

API Flask (serveur ML)

Logstash

Filebeat

Simulateur de logs JSON

Ouverture automatique de Kibana dans le navigateur

📂 Arborescence du projet
graphql
Copier
Modifier
PipelineELK_ML_LocalV1/
│
├── ml_api/
│   └── app.py             # API Flask avec modèle ML
│
├── simulateur_logs/
│   └── simulate_logs.py   # Générateur de logs simulés
│
├── ELK_Stack_8.8.0/
│   ├── elasticsearch-8.8.0/
│   ├── kibana-8.8.0/
│   └── logstash-8.8.0/
│       └── config/
│           └── logstash-ml-elastic.conf
│
├── filebeat-8.8.0-windows-x86_64/
│   └── filebeat-logstash-ml.yml
│
├── venv/                  # Environnement virtuel Python
├── start_pipeline.bat     # Script d'exécution global
├── clean_registry.ps1     # Script PowerShell de nettoyage Filebeat
└── requirements.txt
✅ Exemple de prédiction API
bash
Copier
Modifier
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 491,
    "dst_bytes": 0,
    "wrong_fragment": 0,
    "hot": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "count": 2,
    "srv_count": 2,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0
  }'
🚨 Alertes de sécurité
Vous pouvez configurer des règles Kibana ou intégrer Watcher (Elastic) pour générer :

des alertes email ou webhook

des tableaux de bord personnalisés avec des filtres sur prediction == 1

Souhaitez-vous que je vous fournisse directement ce fichier README.md prêt à injecter dans votre dépôt GitHub ?







Vous avez dit :
meme chose avec PipelineELK_ML_DockerV2 la dockeurisation total sans file beats et visualisation en temps réel des logs avec prediction attack ou normal et enfin dashboarding des visualisations donner moi un readme 


ChatGPT a dit :
Voici une version complète et professionnelle du fichier README.md pour votre projet PipelineELK_ML_DockerV2, qui décrit une architecture Dockerisée sans Filebeat, avec visualisation en temps réel des prédictions (attack vs normal) dans Kibana :

📦 PipelineELK_ML_DockerV2 – Détection d'intrusions ML & ELK (Dockerisé)
🎯 Objectif
Ce projet propose une solution dockerisée de bout en bout pour la détection des attaques réseau en temps réel à l’aide :

d’un modèle XGBoost encapsulé dans une API Flask,

d’un pipeline Logstash → ML → Elasticsearch,

de Kibana pour la visualisation dynamique des prédictions.

💡 Contrairement à la version locale, cette architecture n’utilise pas Filebeat, les logs sont injectés directement via un simulateur Dockerisé.

🧰 Technologies utilisées
Composant	Version	Rôle
Docker	Latest	Conteneurisation
Flask	2.3.3	API REST pour les prédictions
XGBoost	3.0.0	Modèle de classification
Logstash	8.8.0	Envoi des logs → appel API Flask → enrichissement
Elasticsearch	8.8.0	Indexation des logs enrichis
Kibana	8.8.0	Visualisation des prédictions en temps réel
Python	3.x	Environnement ML pour l'API et simulateur

🔁 Architecture Dockerisée
mermaid
Copier
Modifier
graph TD
    A[Simulateur de logs] --> B[Logstash]
    B -->|Appel HTTP| C[API Flask (XGBoost)]
    C -->|Résultat prédiction| B
    B --> D[Elasticsearch]
    D --> E[Kibana]
🐳 Conteneurs Docker
Service	Description
ml-api	Contient l’API Flask qui charge le .joblib
logstash	Pipeline qui enrichit les logs avec la prédiction
elasticsearch	Stockage et recherche des logs
kibana	Interface de visualisation web
simulator	Script Python générant des logs JSON

🚀 Lancement automatique
Cloner le dépôt puis exécuter :

bash
Copier
Modifier
docker-compose up --build
Le système :

Génère des logs via simulator

Envoie ces logs au pipeline logstash

Appelle l'API Flask pour obtenir la prédiction (0=normal, 1=attack)

Indexe les résultats dans Elasticsearch

Affiche les logs prédits en temps réel dans Kibana

📁 Structure du projet
bash
Copier
Modifier
PipelineELK_ML_DockerV2/
│
├── docker-compose.yml
├── ml_api/
│   ├── app.py
│   ├── model/
│   │   └── xgb_model.joblib
│   └── Dockerfile
│
├── logstash/
│   ├── Dockerfile
│   └── pipeline.conf
│
├── simulator/
│   ├── simulate_logs.py
│   └── Dockerfile
│
├── elasticsearch/         # Persistance des données
├── kibana/                # Configuration Kibana (optionnel)
└── README.md
🧠 Exemple de log enrichi (post-prédiction)
json
Copier
Modifier
{
  "duration": 0,
  "protocol_type": "tcp",
  "src_bytes": 500,
  "dst_bytes": 0,
  ...
  "prediction": 1
}
📊 Dashboard Kibana
Créez un tableau de bord incluant :

Histogramme du nombre d’attaques (prediction == 1)

Pie chart : ratio attaques / trafic normal

Table : top protocoles/services associés aux attaques

⚠️ Alerte (optionnel)
Vous pouvez activer des alertes dans Kibana (via Stack Management → Rules) pour détecter en temps réel :

un seuil de prédictions 1 sur une fenêtre temporelle

un type d’attaque récurrent par IP/source

