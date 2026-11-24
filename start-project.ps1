$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop and Docker Compose are required.'
}

if (-not (Test-Path -LiteralPath '.env')) {
    $template = Get-Content -LiteralPath '.env.example' -Raw
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    $secret = ([BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
    $template = $template.Replace('ADMIN_TOKEN=replace-with-a-long-random-secret', "ADMIN_TOKEN=$secret")
    [IO.File]::WriteAllText((Join-Path $PSScriptRoot '.env'), $template)
    Write-Host 'Created ignored .env with a random admin token.'
}

docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Compose could not start the stack. Check docker compose logs.'
}
Write-Host 'API: http://127.0.0.1:8000/docs'
Write-Host 'Phoenix: http://127.0.0.1:6006'
