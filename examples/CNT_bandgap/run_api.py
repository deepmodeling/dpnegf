"""Python-API tour of the band-gap calculator on a (7,0) carbon nanotube.

Runs the same task as ``input.json`` but through the library interface.  When
the CLI example has been run first, the H/S block cache and matching per-k
spectra are reused: no model build and no re-solve happens for cached points.
"""

from pathlib import Path

from dpnegf.runner.cal_bandgap import BandGapCalculator

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "../long_cnt/nnenv.iter201150.pth"
STRUCTURE = HERE / "cnt7_0_len6.xyz"
# Point the two-level cache at the CLI run's results directory; falls back to
# a cold model inference when the files do not exist yet.
CACHE = HERE / "output_cli/results/hs_blocks.npz"

calculator = BandGapCalculator(
    structure=str(STRUCTURE),
    checkpoint=str(CHECKPOINT),
    dtype="float32",
    task_options={
        "structure_options": {"mode": "periodic", "pbc": [False, False, True]},
        "occupation_options": {"nel_atom": {"C": 4.0}},
        "solver_options": {"method": "shift-invert", "energy_shift": None},
        "cache_options": {"path": str(CACHE)},
    },
    results_path=str(HERE / "output_api/results"),
)
result = calculator.compute()

print(f"gap_status       : {result['gap_status']}")
print(f"signed gap       : {result['signed_gap_eV']:.4f} eV (nonnegative {result['band_gap_eV']:.4f} eV)")
print(f"min direct gap   : {result['minimum_direct_gap_eV']:.4f} eV")
print(f"VBM              : {result['vbm_eV']:.4f} eV at k={result['vbm_k_fractional']}")
print(f"CBM              : {result['cbm_eV']:.4f} eV at k={result['cbm_k_fractional']}")
print(f"converged        : {result['converged']} (tol {result['requested_resolution_eV']} eV)")
print(f"solver           : {result['solver']}")
print(f"H/S cache hit    : {result['cache']['hit']} "
      f"(spectrum hits {result['cache']['spectrum_hits']}, misses {result['cache']['spectrum_misses']})")
print("k convergence    :")
for level in result["k_convergence"]:
    change = level["max_edge_change_eV"]
    print(f"  nk={level['nk']:>3}  VBM={level['vbm_eV']: .4f}  CBM={level['cbm_eV']: .4f}"
          f"  dE={change if change is not None else float('nan'): .4f}  converged={level['converged']}")
