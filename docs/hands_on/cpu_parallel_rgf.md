# CPU Parallel RGF

Set `task_options.rgf_options.cpu_workers` above one to distribute a non-SCF
energy sweep across CPU processes:

```json
{
  "device": "cpu",
  "task_options": {
    "task": "negf",
    "scf": false,
    "rgf_options": {
      "device": "cpu",
      "cpu_workers": 4,
      "cpu_threads": 4,
      "e_batch_size": 1,
      "startup_timeout": 600.0,
      "sync_timeout": 600.0,
      "task_timeout": 3000.0,
      "max_pending_results": null
    }
  }
}
```

`cpu_workers: 1` keeps RGF in the main process. With multiple workers, each
process uses one energy point per task; `e_batch_size` may be omitted or set to
1. Batched CPU RGF is intentionally disabled in this mode because this build
has encountered a non-returning MKL `ZLASWP` operation for multi-threaded
batched complex solves.

`cpu_threads` limits Torch intra-op and loaded BLAS/OpenMP pools in the main
process during serial RGF. In a worker pool it sets threads per worker. Null
preserves the caller's limits for a single process and selects one thread per
worker when `cpu_workers` is greater than one. The requested worker/thread
product must fit `joblib.cpu_count()`, which accounts for the process CPU
affinity and cgroup limits. This is a resource ceiling, not a recommendation to
use every available core. CPU RGF settings cannot be combined with CUDA or an
explicit `devices` list, and multiple workers require `scf: false`.

The parent prepares H/S for each k point and applied potential, then publishes
the six block groups in one aligned shared-memory region. Workers attach NumPy
and Torch views without copying. A changed k point, potential, or block layout
is published as a new version only after the previous energy sweep has drained;
workers acknowledge the new mapping before the parent unlinks the old region.
Published H/S blocks must not be modified in place. Shared memory holds H/S
only; each worker still owns its self-energy matrices, Green functions,
workspaces and outputs.

Workers claim one energy task at a time. Replies can arrive out of order, while
the parent stores and merges them in the original energy order. The default
`max_pending_results` is twice the worker count and bounds active plus completed
but not yet merged batches. This is a batch-count limit, not an output-memory
budget.

Worker readiness and synchronization have 600-second deadlines by default.
Each task has a 3000-second deadline covering cache I/O, RGF, output handling,
serialization and IPC. Set an individual timeout to null to disable that
deadline, or raise it for a large system or slow filesystem. These limits do not
cover parent-side self-energy preparation, H/S loading and publication, or a
whole NEGF run. A parent-side command send can also block before a worker reply
deadline can be checked. A timeout terminates all workers and discards the
incomplete sweep; no energy point is skipped or retried.

At `INFO` level, each worker reports only its PID, device and Torch thread
counts. Run the CLI with `-ll DEBUG` to include per-energy RGF chunk progress
and the full BLAS/OpenMP `threadpool_info()` diagnostics.

Python callers must use a main guard because workers use the `spawn` start
method:

```python
from dpnegf.entrypoints.run import run

if __name__ == "__main__":
    run(INPUT, MODEL, STRUCTURE, OUTPUT, LOG_PATH)
```

For memory comparisons, use process-tree PSS or USS when available. RSS counts
shared pages in every process and overstates physical H/S duplication.
