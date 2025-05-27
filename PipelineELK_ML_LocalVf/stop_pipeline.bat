@echo off
setlocal
@REM Ferme toutes les fenêtres démarrées avec un titre (start "TITRE") pour : API Flask, Logstash, Kibana, Elasticsearch, Filebeat, Simulateur.

echo === [1] Fermeture des fenêtres nommées de la stack ===
for %%T in ("Elasticsearch" "Kibana" "API Flask" "Logstash" "Filebeat" "Simulateur Logs") do (
    echo. - Fermeture de la fenêtre %%~T
    taskkill /FI "WINDOWTITLE eq %%~T" /T /F >nul 2>&1
)
@REM Termine les processus restants au cas où certaines fenêtres seraient fermées manuellement.

echo === [2] Arrêt des processus résiduels (sécurité) ===
for %%P in (elasticsearch.bat kibana.bat python.exe logstash.bat filebeat.exe) do (
    echo. - Arrêt de %%~P
    taskkill /IM %%~P /T /F >nul 2>&1
)

echo ✅ Tous les composants ont été arrêtés.
pause >nul
endlocal
