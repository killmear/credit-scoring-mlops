$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    docker build -t study-credit-scoring .
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка сборки Docker' }
    docker run --rm -p 127.0.0.1:8000:8000 --name study-credit-scoring study-credit-scoring
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка запуска Docker' }
} finally {
    Pop-Location
}
