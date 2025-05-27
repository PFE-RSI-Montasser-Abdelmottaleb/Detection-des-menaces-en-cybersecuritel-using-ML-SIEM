# clean_registry.ps1
$registryPath = "D:\pfe_iset\IntegrationML_ELK\ELK_Stack_8.8.0\filebeat-8.8.0-windows-x86_64\data\registry"
if (Test-Path $registryPath) {
    Remove-Item "$registryPath\*" -Recurse -Force
    Write-Output "✅ Dossier registry supprimé."
} else {
    Write-Output "❌ Dossier registry introuvable."
}
