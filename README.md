# Lightbaker

Indoor showroom lightmap baker: atlas charting, deterministic path-traced
lighting, distributed block tasks, and a browser task panel.

## Graphical entry point (primary)

```sh
python3 -m lightbaker.cli server            # http://127.0.0.1:8000/
# or: python3 -m lightbaker.server
```

The panel is the normal way to bake: choose the scene, atlas size, texel
density, quality preset and workers, press **启动烘焙**, then inspect the
lightmap, block boundaries, coverage mask, verification and the 3D runtime
preview. Failed or stale blocks can be retried without recomputing finished
ones. Existing job folders placed in the workspace are adopted automatically,
so a legacy job can be opened, converted and completed from the panel.

## Command line (auxiliary)

The CLI is for automation and reproducibility checks only; it uses the same
protocol, build lock and conversion gate as the panel.

```sh
python3 -m lightbaker.cli scenes
python3 -m lightbaker.cli bake --scene closed_room --atlas 256 --density 12 \
    --quality medium --out out/my_job
python3 -m lightbaker.cli demo --out out/demo
```

## Task/result protocol (v2)

Every task and result carries explicit identity:

* **atlas** — revision digest over chart segmentation, per-texel position /
  normal / face / barycentric and coverage, plus scene, size, texel density
  and block size.
* **coverage** — the exact sorted list of real (non-dilation) texel indices a
  task owns, its count and digest, and the per-texel sample budget.
* **compute** — kernel revision, samples, effective samples, seed, bounce
  depth, block size, units and seed scheme.
* **task_id** — digest over atlas revision + block rect + coverage digest +
  compute descriptor.

A block's `(bx, by)` coordinates are therefore never its identity: the same
rectangle under a different atlas revision, density or sample configuration is
a different task (`tests/test_core.py::test_task_identity_not_block_coordinate`).

The worker rebuilds the atlas from the task's own parameters, checks the
revision and exact texel set, bakes, validates the result against the task and
writes it atomically. The receiver revalidates every result before assembly,
so a stale or mixed result makes the job `partial` rather than contributing
wrong energy. `verification.json` proves that every real texel was baked
exactly once and that block rectangles lie on the atlas grid.

## Legacy result conversion (v1 → v2)

Legacy block payloads have no atlas identity. They are reused only when a
*complete* conversion check passes:

* legacy `manifest.json` / `config.json` exists and its scene, atlas size,
  texel density, block size, samples, seed and bounce depth equal the task's;
* the block index and rect equal the task's;
* the value keys equal the task's exact real-texel set (count and membership);
* units, seed scheme and declared ray count are consistent.

Every rejection is recorded with its reason in `migration_report.json`; no
field is filled in from a default. Rejected blocks are re-baked on the real
worker path. Original v1 payloads are archived under `blocks/legacy/`.

## Build lock

`bake.lock.json` pins the protocol version, result/atlas schemas, migration
label, kernel revision, units and seed scheme. The panel and CLI refuse to
bake when the lock does not match the code. Change the lock together with the
protocol version and kernel revision.

## Tests

```sh
python3 -m lightbaker.tests.test_core
```

The suite covers determinism, no chart bleed, no light leak, real indirect
bounces, task identity under equal block coordinates, the legacy conversion
gate (accept and reject cases), and an end-to-end re-bake that verifies full
atlas coverage.
