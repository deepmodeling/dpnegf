# CNT band-gap example

Band-gap calculation of a (7,0) zigzag carbon nanotube with the `cal_bandgap`
task: a one-dimensional periodic supercell (168 atoms, 2184 orbitals from the
DeePTB e3 checkpoint) solved on CPU with the verified shift-invert solver.

The full structure is kept as provided — no unit-cell reduction, no end
clipping — and the model is evaluated once; all k points reuse the same
real-space H/S blocks through the two-level cache.

## Files

| file | role |
| :--- | :--- |
| `cnt7_0_len6.xyz` | (7,0) CNT, 6 translational periods (168 atoms), seamless z cell 25.56 Å, `pbc="F F T"` |
| `input.json` | CLI configuration for `dpnegf run` (shift-invert, default nested k grids 9 → 17 → 33 → 65) |
| `run_api.py` | the same calculation through the `BandGapCalculator` Python API |
| `output_cli/` | CLI outputs (created by the run) |
| `output_api/` | API outputs (created by `run_api.py`) |

## Supported systems

`cal_bandgap` is limited to **finite structures** (HOMO–LUMO gap) and
**one-dimensional periodic structures** (a single periodic axis, as in this
example). Two- or three-dimensional periodic boundary conditions are rejected
with an explicit error; arbitrary 2D/3D k-space search is out of scope for
this first version. Magnetic/SOC models, odd-electron counts and GPU solving
are  not supported yet. Inference runs on CPU in **float32 only**: DeePTB
checkpoints are trained in float32, so requesting float64 (or loading a
float64 checkpoint) raises an explicit error.

## input.json core options

Note: `dpnegf run` parses strict JSON (`json.load`), so the shipped
`input.json` **cannot** contain comments. The annotated snippets below are
for explanation only; field values match `input.json`.

```jsonc
"structure_options": {
  "mode": "periodic",        // structure mode: auto / finite / periodic; here: 1D-periodic supercell
  "pbc": [false, false, true],  // periodic boundary flags [x, y, z]; at most one may be true
                                // (1D only; multi-axis PBC is rejected pending a future release)
  "periodic_length": null    // optional: replace the axial lattice-vector length (Å) without scaling
                             // coordinates; null keeps the cell from the structure file
},
"occupation_options": {
  "nel_atom": {"C": 4.0},    // valence electrons per element; sets the occupied band count m = N_e/2
  "charge": 0.0              // total charge; positive values remove electrons (occupation count only, no SCF)
},
"solver_options": {
  "method": "shift-invert",  // spectrum solver: auto / dense / inertia / shift-invert
  "backend": "superlu",      // sparse factorization backend (only superlu is supported)
  "dense_limit": 1000,       // with method=auto, dimensions <= this use dense, larger use inertia
  "edge_tol": 1e-05,         // energy interval width of the inertia bisection (eV)
  "energy_bounds": [-10.0, 10.0],  // initial energy interval for inertia counts (eV); expands when needed
  "energy_shift": null       // shift-invert sigma hint (eV); null = coarse location on the first k,
                             // warm start from the nearest solved k afterwards
},
"k_options": {
  "initial_nk": 9,           // initial half-BZ nested grid (>= 3); grids refine as n -> 2(n-1)+1
  "min_nk": 17,              // convergence is not accepted below this grid size (avoids false
                             // convergence on coarse grids)
  "max_nk": 65,              // maximum grid size; 9 -> 17 -> 33 -> 65
  "gap_tol": 0.01,           // adjacent-grid band-edge stability criterion (eV); must exceed 2*edge_tol
  "local_refine": false      // refine sampled extrema with additional real k-point evaluations
},
"output_options": {
  "bands_per_side": 4,       // number of bands kept on each side of the occupation edge
  "all_bands": false,        // keep the full spectrum (dense solver only, expensive for large systems)
  "plot": true,              // write bands.png / gap.png (no display server required)
  "plot_nk": 0,              // extra evenly spaced k points for plotting (0 = none); new points
                             // re-summarize the gap
  "plot_window": null        // plotted energy window [min, max] (eV) relative to the mid-gap reference;
                             // null = all retained bands
}
```

## Requirements

The DeePTB e3 checkpoint for carbon is expected at
`examples/long_cnt/nnenv.iter201150.pth` (about 21 MB). Pass a different path
with `-i` / the `checkpoint=` argument if your copy lives elsewhere.

## Run with the CLI

```bash
dpnegf run input.json \
  -i ../long_cnt/nnenv.iter201150.pth \
  -stu cnt7_0_len6.xyz \
  -o output_cli
```

The first run performs the e3 inference and solves the nested k grids; on this
structure each sparse solve takes tens of seconds on CPU, so expect a few
minutes in total (about 6 minutes for the default run: inference plus the
9-point and 17-point grids). Reruns hit the two-level cache
(`output_cli/results/hs_blocks.npz` plus per-k spectra) and skip both the model
build and already solved k points.

The log reports the resolved spectrum solver and orbital count, then tracks
the adaptive k search level by level — grid size, gap, band-edge and gap
changes versus the previous grid — and finishes with the final gap summary:

```text
cal_bandgap: 2184 orbitals, 672 electrons, 336 occupied bands; solving bands 333..340; spectrum solver: shift-invert.
cal_bandgap k-grid nk=9: gap=0.598129 eV (VBM=0.720946, CBM=1.319075 eV); first grid level.
cal_bandgap k-grid nk=17: gap=0.598129 eV; band-edge change 0.000000 eV, gap change 0.000000 eV -> converged (tolerance 0.01 eV).
cal_bandgap finished: positive_gap, converged=True.
  band gap = 0.598129 eV
  VBM = 0.720946 eV at k=[0.0, 0.0, 0.0], CBM = 1.319075 eV at k=[0.0, 0.0, 0.0]
```

## Run with the Python API

```bash
python run_api.py
```

`run_api.py` points its cache at the CLI output directory, so after the CLI
run it starts from the cached H/S blocks (no model construction) and reuses
all 17 cached spectra — it finishes in a few seconds and writes its own
outputs under `output_api/results`.

## Expected results

For the (7,0) CNT the calculator reports a direct gap at Γ:

- `gap_status = positive_gap`, `signed_gap_eV ≈ 0.5981` (VBM ≈ 0.7209 eV,
  CBM ≈ 1.3191 eV, both at `k = [0, 0, 0]`);
- `converged = true` at the 17-point grid: VBM/CBM are unchanged between the
  9- and 17-point grids (flat band edges at Γ), and `min_nk = 17` is the
  first grid allowed to accept the criterion, recorded in `k_convergence`;
- solver entries per k point are `shift-invert`, with the inertia-probe count
  in `k_spectra` showing the warm start (the first k point needs a coarse
  shift location, later k points only need the window-verification probes).

Outputs (under `OUTPUT/results/`): `gap.json` (headline values and
diagnostics), `bands.npz` / `bands.csv` (near-gap spectra per k),
`band_edges.csv`, and `bands.png` / `gap.png` plots — `gap.png` is annotated
with the final gap value, status, and the VBM/CBM positions.

See `docs/cal_bandgap.md` for the full option schema, solver descriptions and
numerical limitations.
