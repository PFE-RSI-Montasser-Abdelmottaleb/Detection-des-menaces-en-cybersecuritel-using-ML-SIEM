# Détection des Menaces en Cybersécurité avec Machine Learning et Dataset NSL-KDD 🛡️

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
