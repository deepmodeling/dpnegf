#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

echo "--- Installing/updating package from PR in editable mode ---"

# We use 'conda run' to execute the commands within the 'dpusk' environment.
# 1. `pip install -e .`: The '-e' (editable) flag is crucial. It installs the
#    package from the current directory (the PR's code) in a way that links
#    back to the source files. This ensures that the tests run against the
#    very latest code from the pull request, not the version baked into the
#    Docker image.
# 2. `pytest ./tests/`: After the package is installed, we run the tests.

conda run -n dpnegf bash -c "
  pip install -e .

  TORCH_VERSION=\$(python -c \"import torch; print(torch.__version__.split('+')[0])\")
  TORCH_BACKEND=\$(python -c \"import torch; print('cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', ''))\")
  TORCH_SCATTER_INDEX=\"https://data.pyg.org/whl/torch-\${TORCH_VERSION}+\${TORCH_BACKEND}.html\"
  pip install --no-cache-dir --force-reinstall --only-binary=torch_scatter torch_scatter -f \"\${TORCH_SCATTER_INDEX}\"

  pytest dpnegf/tests/
"

echo "--- Unit Tests Passed Successfully ---"
