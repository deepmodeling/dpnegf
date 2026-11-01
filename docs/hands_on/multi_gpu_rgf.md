# Multi-GPU RGF

Set `task_options.rgf_options.devices` to distribute a non-SCF energy-grid
sweep over multiple GPUs:

```json
{
  "device": "cpu",
  "task_options": {
    "task": "negf",
    "scf": false,
    "rgf_options": {
      "devices": ["cuda:0", "cuda:1"],
      "e_batch_size": 1
    }
  }
}
```

Merge these settings into a complete NEGF input. The top-level `device` controls
model inference and Hamiltonian initialization independently; it can also be
`"cuda:0"`.

`devices` overrides `rgf_options.device`. Its indices refer to the GPUs visible
to the process. For example, with `CUDA_VISIBLE_DEVICES=2,3`, `cuda:0` and `cuda:1`
refer to physical GPUs 2 and 3. Omitting `devices` preserves the single-device
behavior, including when `device` is `"cuda"`.

`e_batch_size` is the number of energy points in each batch on each GPU. With
two GPUs and a batch size of 1, energies are assigned alternately: GPU 0 computes
points 0, 2, 4, ... and GPU 1 computes points 1, 3, 5, .... For a batch size of 4,
GPU 0 first computes points 0–3 and GPU 1 computes points 4–7. The next batch
assigned to a GPU starts after its previous batch has completed. At most one
batch is active on each GPU. A final short batch is supported.

Omitting `e_batch_size` or setting it to `null` selects a common batch size from
the smallest free-memory budget across the chosen GPUs, capped to distribute
work over all GPUs when there are enough energy points. An explicit batch size
large enough to contain the whole grid creates only one batch.

Each selected GPU has one independent process, created with Python's `spawn`
start method. The process constructs a model-independent RGF solver from cache
paths and numerical settings; the neural model and `NEGF` runner stay in the
parent process. CUDA contexts and runtime state belong to each worker process.
Each worker caches its own H/S blocks. Self-energy preparation runs before the
parallel sweep, and workers read the resulting CPU caches. Transmission, DOS,
LDOS and transmission eigenvalues/channels are merged in the original energy
order. Eigenchannel files retain their global energy indices. Conductance and
current derived from the transmission use the merged spectrum.

Here, the H/S cache means the saved `HS_device.h5` (or legacy `HS_device.pth`)
files on disk. Each worker opens these files itself when it first needs a k
point's H/S. The parent passes the cache path and metadata; it does not send
its in-memory H/S tensors to workers. After reading the files, a worker keeps
its own H/S blocks for subsequent energy batches at that k point.

The same worker pool serves all k points in a calculation, retaining each
worker's PID and CUDA context. H/S blocks are refreshed when the k point changes.
Workers shut down after the sweep. If a worker reports an exception or exits
unexpectedly, the parent terminates and joins every worker, including workers
still blocked in a solver, and raises an error with the available worker
traceback.

When calling the Python API from a script, put the entry call inside a main
guard so spawned processes can import the script safely:

```python
from dpnegf.entrypoints.run import run

if __name__ == "__main__":
    run(INPUT, MODEL, STRUCTURE, OUTPUT, LOG_PATH)
```

The `dpnegf` command-line entry point already provides this guard.

This mode currently requires `scf: false`. Density and local-current integrals
continue to use the first selected GPU. It distributes energies within each k
point; k points are processed sequentially. This mode does not split a single
energy calculation or its H/S storage across GPUs.
