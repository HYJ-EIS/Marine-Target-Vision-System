$ErrorActionPreference = "Stop"

# Edit these values once for your machine.
$RepoDir = "D:\APPS\X-AnyLabeling"
$CondaEnvName = "x-anylabeling"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Initialize-Conda {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        throw "未找到 conda。请先确保 Anaconda/Miniconda 已安装并已加入 PATH。"
    }

    $hook = conda shell.powershell hook | Out-String
    if (-not $hook) {
        throw "无法初始化 conda PowerShell hook。"
    }
    Invoke-Expression $hook
}

Write-Step "检查仓库目录"
if (-not (Test-Path -LiteralPath $RepoDir)) {
    throw "仓库目录不存在: $RepoDir"
}

Set-Location -LiteralPath $RepoDir

Write-Step "激活 Conda 环境"
Initialize-Conda
conda activate $CondaEnvName

Write-Step "启动 X-AnyLabeling"
xanylabeling
