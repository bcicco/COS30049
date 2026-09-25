# PROJECT SET UP: 

## Requirements: 

### Astral UV 
#### installation: 
MAC: 
curl -LsSf https://astral.sh/uv/install.sh | sh

POWERSHELL: 
Set-ExecutionPolicy Bypass -Scope Process -Force
irm https://astral.sh/uv/install.ps1 | sh

#### sync venv: 
uv sync

#### add packages to venv: 
edit dependencies in pyproject.toml

