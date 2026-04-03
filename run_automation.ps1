$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

function Get-LauncherCommand {
    if (Test-Path -LiteralPath ".venv\Scripts\python.exe") {
        return @(".venv\Scripts\python.exe", "run_automation.py")
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "run_automation.py")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python", "run_automation.py")
    }

    Write-Host "[Launcher] Python bulunamadi. Lutfen Python 3.10+ kur."
    exit 1
}

$command = Get-LauncherCommand
$executable = $command[0]
$arguments = @()
if ($command.Count -gt 1) {
    $arguments = $command[1..($command.Count - 1)]
}

try {
    & $executable @arguments
    if ($null -ne $LASTEXITCODE) {
        exit ([int]$LASTEXITCODE)
    }
    exit 0
}
catch [System.Management.Automation.PipelineStoppedException] {
    exit 130
}
catch {
    Write-Host ""
    Write-Host ("[Launcher] Hata: " + $_.Exception.Message)
    exit 1
}
