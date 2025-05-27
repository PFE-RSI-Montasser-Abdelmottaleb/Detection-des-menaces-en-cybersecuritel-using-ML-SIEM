@echo off
echo [*] Création de l'environnement virtuel...
python -m venv venv

echo [*] Activation de l'environnement virtuel...
call venv\Scripts\activate

echo [*] Mise à jour de pip...
python -m pip install --upgrade pip

echo [*] Installation des dépendances...
pip install Flask==2.3.3 ^
            xgboost==3.0.0 ^
            scikit-learn==1.3.2 ^
            joblib==1.3.2 ^
            requests==2.31.0 ^
            Werkzeug==2.3.7

echo [✔] Environnement prêt.
pause
