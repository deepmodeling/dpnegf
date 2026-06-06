#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

echo "--- [ENV] Torch and Scatter ---"
# 1. 打印 pip 识别到的安装信息和路径
conda run -n dpnegf bash -c "pip show torch torch-scatter || true"
# 2. 尝试直接 import 并打印底层版本，如果这里报错，说明 Docker 镜像本身就有问题
conda run -n dpnegf bash -c "python -c \"import torch; print('Torch:', torch.__version__); import torch_scatter; print('Scatter:', torch_scatter.__version__)\" || echo '❌ Import 失败，镜像底层环境已损坏'"

echo "--- Installing/updating package from PR in editable mode ---"

# We use 'conda run' to execute the commands within the 'dpnegf' environment.
# 1. `pip install -e .`: The '-e' (editable) flag is crucial. It installs the
#    package from the current directory (the PR's code) in a way that links
#    back to the source files. This ensures that the tests run against the
#    very latest code from the pull request, not the version baked into the
#    Docker image.
# 2. `pytest ./tests/`: After the package is installed, we run the tests.

conda run -n dpnegf bash -c "pip install -e . 'torch==2.1.1' --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-2.1.1+cpu.html --only-binary=torch-scatter && pytest dpnegf/tests/"

echo "--- Unit Tests Passed Successfully ---"

