@echo off
setlocal

echo === [1] Nettoyage du dossier registry de Filebeat ===
powershell -ExecutionPolicy Bypass -File "D:\pfe_iset\IntegrationML_ELK\ml_pipeline_local\clean_registry.ps1"

echo === [2] Démarrage d'Elasticsearch ===
start "Elasticsearch" "D:\pfe_iset\IntegrationML_ELK\ELK_Stack_8.8.0\elasticsearch-8.8.0\bin\elasticsearch.bat"
timeout /t 20 >nul

echo === [3] Démarrage de Kibana ===
start "Kibana" "D:\pfe_iset\IntegrationML_ELK\ELK_Stack_8.8.0\kibana-8.8.0\bin\kibana.bat"
timeout /t 100 >nul

echo === [4] Démarrage de l'API Flask ===
start "API Flask" cmd /k "cd /d D:\pfe_iset\IntegrationML_ELK\ml_pipeline_local\ml_api && call D:\pfe_iset\IntegrationML_ELK\venv\Scripts\activate && python app.py"
timeout /t 10 >nul

echo === [5] Démarrage de Logstash ===
start "Logstash" cmd /k "cd /d D:\pfe_iset\IntegrationML_ELK\ELK_Stack_8.8.0\logstash-8.8.0 && bin\logstash.bat -f config\logstash-ml-elastic.conf --config.reload.automatic""
timeout /t 30 >nul

echo === [6] Démarrage de Filebeat ===
start "Filebeat" cmd /k "cd /d D:\pfe_iset\IntegrationML_ELK\ELK_Stack_8.8.0\filebeat-8.8.0-windows-x86_64 && filebeat.exe -e -c filebeat-logstash-ml.yml -d publish"
timeout /t 5 >nul

echo === [7] Lancement du simulateur de logs ===
start "Simulateur Logs" cmd /k "cd /d D:\pfe_iset\IntegrationML_ELK\ml_pipeline_local\simulateur_logs && call D:\pfe_iset\IntegrationML_ELK\venv\Scripts\activate && python simulate_logs.py"
timeout /t 5 >nul

echo === [8] Ouverture de Kibana dans le navigateur ===
start http://localhost:5601

echo ✅ Stack ELK ML opérationnelle. Appuyez sur une touche pour quitter ce script...
pause >nul
endlocal
