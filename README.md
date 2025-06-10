# I Détection Intelligente des Intrusions en Cybersécurité avec Machine Learning et Dataset NSL-KDD 

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

######################################################################################################

# III PipelineELK_ML_DockerV2 – Détection d'intrusions ML & ELK (Dockerisé)
# Objectif
Ce projet propose une solution dockerisée de bout en bout pour la détection des attaques réseau en temps réel à l’aide :

1. d’un modèle XGBoost encapsulé dans une API Flask,

2. d’un pipeline Logstash → ML → Elasticsearch,

3. de Kibana pour la visualisation dynamique des prédictions.

Rq : Contrairement à la version locale, cette architecture n’utilise pas Filebeat, les logs sont injectés directement via un simulateur Dockerisé.

 # Technologies utilisées
 
Composant	Version	Rôle

Docker	Latest	Conteneurisation

Flask	2.3.3	API REST pour les prédictions

XGBoost	3.0.0	Modèle de classification

Logstash	8.8.0	Envoi des logs → appel API Flask → enrichissement

Elasticsearch	8.8.0	Indexation des logs enrichis

Kibana	8.8.0	Visualisation des prédictions en temps réel

Python	3.x	Environnement ML pour l'API et simulateur

# Architecture Dockerisée

    A[Simulateur de logs] --> B[Logstash]
    B -->|Appel HTTP| C[API Flask (XGBoost)]
    C -->|Résultat prédiction| B
    B --> D[Elasticsearch]
    D --> E[Kibana]

    
# Conteneurs Docker

Service	Description

1.ml-api	Contient l’API Flask qui charge le .joblib

2.logstash	Pipeline qui enrichit les logs avec la prédiction

3.elasticsearch	Stockage et recherche des logs

4.kibana	Interface de visualisation web

5.simulator	Script Python générant des logs JSON

🚀 Lancement automatique
Cloner le dépôt puis exécuter :

docker-compose up --build

Le système :

1. Génère des logs via simulator

2. Envoie ces logs au pipeline logstash

3. Appelle l'API Flask pour obtenir la prédiction (0=normal, 1=attack)

4. Indexe les résultats dans Elasticsearch

5. Affiche les logs prédits en temps réel dans Kibana


# Dashboard Kibana

Créez un tableau de bord incluant :

Histogramme du nombre d’attaques (prediction == 1)

Pie chart : ratio attaques / trafic normal

Table : top protocoles/services associés aux attaques

#Alerte 

activer des alertes dans Kibana (via Stack Management → Rules) pour détecter en temps réel :

un seuil de prédictions 1 sur une fenêtre temporelle



