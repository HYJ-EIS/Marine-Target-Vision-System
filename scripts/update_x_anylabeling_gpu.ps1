$ErrorActionPreference = "Stop"

# Edit these values once for your machine.
$RepoDir = "D:\APPS\X-AnyLabeling"
$CondaEnvName = "x-anylabeling"
$ProxyUrl = "http://127.0.0.1:7892"

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

Write-Step "配置 Git 代理"
git config --global http.proxy $ProxyUrl
git config --global https.proxy $ProxyUrl
$env:http_proxy = $ProxyUrl
$env:https_proxy = $ProxyUrl

Write-Step "拉取最新代码"
git pull

Write-Step "初始化 Conda"
Initialize-Conda

Write-Step "创建或激活环境"
$envExists = conda env list | Select-String -SimpleMatch $CondaEnvName
if (-not $envExists) {
    conda create -n $CondaEnvName python=3.12 -y
}
conda activate $CondaEnvName

Write-Step "更新安装工具"
python -m pip install --upgrade pip
python -m pip install --upgrade uv

Write-Step "清理旧包冲突"
python -m pip uninstall anylabeling -y

Write-Step "安装 X-AnyLabeling GPU 依赖"
uv pip install -e ".[gpu]"

Write-Step "验证安装"
xanylabeling version
xanylabeling checks

Write-Step "更新完成"
Write-Host "仓库目录: $RepoDir" -ForegroundColor Green
Write-Host "Conda 环境: $CondaEnvName" -ForegroundColor Green
