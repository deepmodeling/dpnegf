# DPNEGF

**DPNEGF** is a Python package that integrates the Deep Learning Tight-Binding (**DeePTB**) approach with the Non-Equilibrium Green’s Function (**NEGF**) method, establishing an efficient quantum transport simulation framework **DeePTB-NEGF** with first-principles accuracy. 

By using DeePTB-SK or DeePTB-E3—both available within the DeePTB package—DeePTB-NEGF can compute quantum transport properties in open-boundary systems with either environment-corrected **Slater-Koster TB Hamiltonian** or **linear combination of atomic orbitals (LCAO) Kohn-Sham Hamiltonian**.


For more details, see our papers:
  1. [DPNEGF: npj Comput Mater 11, 375 (2025)](https://www.nature.com/articles/s41524-025-01853-6)
  2. [DeePTB-SK: Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
  3. [DeePTB-E3: ICLR 2025 Spotlight](https://openreview.net/forum?id=kpq3IIjUD3)


## Installation

DPNEGF runs inside the DeePTB virtual environment. We use [UV](https://github.com/astral-sh/uv) as the package manager.

- **Requirements**
  - Git
  - Python 3.9 to 3.12 (UV can auto-install if needed)
  - [DeePTB](https://github.com/deepmodeling/DeePTB) ≥ 2.1.1

- **Step 1: Install UV** (if not already installed)
  ```bash
  # On macOS and Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Or using pip
  pip install uv

  # On Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **Step 2: Install DeePTB**
  ```bash
  git clone https://github.com/deepmodeling/DeePTB.git
  cd DeePTB
  uv sync  # Creates .venv and installs DeePTB with all dependencies
  ```
  For GPU support, see [DeePTB README](https://github.com/deepmodeling/DeePTB#installation).

- **Step 3: Add DPNEGF to the DeePTB environment**
  ```bash
  # Clone the DPNEGF repository (you can clone it anywhere)
  git clone https://github.com/deepmodeling/dpnegf.git
  
  # Still inside the DeePTB directory
  uv add /path/to/dpnegf
  ```
  Replace `/path/to/dpnegf` with the actual path to your cloned DPNEGF repository.

- **Run DPNEGF**
  ```bash
  # UV automatically activates the environment
  uv run dpnegf --help

  # Or activate manually
  source .venv/bin/activate  # On Unix/macOS
  .venv\Scripts\activate     # On Windows
  dpnegf --help
  ```
## Test code 

To ensure the code is correctly installed, please run the unit tests first:
```bash
pytest ./dpnegf/tests/
```
Be careful if not all tests pass!


## How to cite

The following references are required to be cited when using DPNEGF. Specifically:

- **For DPNEGF:**
  
    J. Zou, Z. Zhouyin, D. Lin, Y. Huang, L. Zhang, S. Hou and Q. Gu, Deep Learning Accelerated Quantum Transport Simulations in Nanoelectronics: From Break Junctions to Field-Effect Transistors, npj Comput Mater 11, 375 (2025).


- **For DeePTB-SK:**

    Q. Gu, Z. Zhouyin, S. K. Pandey, P. Zhang, L. Zhang, and W. E, Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with Ab Initio Accuracy, Nat Commun 15, 6772 (2024).
  
- **For DeePTB-E3:**
  
    Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, Learning Local Equivariant Representations for Quantum Operators, In The 13th International Conference on Learning Representations (ICLR) 2025. 
