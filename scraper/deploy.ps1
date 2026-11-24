# Run on Windows (PowerShell): copies kfcollect.zip to the server and updates it in one step.
#   powershell -ExecutionPolicy Bypass -File .\deploy.ps1 -Server azureuser@<server>
#   powershell -ExecutionPolicy Bypass -File .\deploy.ps1 -Server azureuser@<server> -Key C:\path\to\key.pem
param(
  [Parameter(Mandatory = $true)][string]$Server,
  [string]$Zip = (Join-Path $PSScriptRoot "kfcollect.zip"),
  [string]$Key
)
if (-not (Test-Path $Zip)) { Write-Error "Can't find $Zip"; exit 1 }
$opts = @()
if ($Key) { $opts += @("-i", $Key) }
scp @opts $Zip "${Server}:kfcollect.zip"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ssh -t @opts $Server 'set -e; rm -rf ~/kfcollect; python3 -m zipfile -e ~/kfcollect.zip ~ && chmod +x ~/kfcollect/*.sh && sudo ~/kfcollect/update.sh'
exit $LASTEXITCODE
