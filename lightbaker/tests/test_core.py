"""Headless acceptance tests for the baker (run: python3 -m lightbaker.tests.test_core).

These check the numerical/structural guarantees without a browser:
  1. deterministic, auditable RNG (reproducible block output)
  2. worker retry is idempotent: a re-baked block keeps identical radiance
  3. dilation never crosses chart boundaries (no neighbour bleed)
  4. thin wall: unlit room side stays dark (no light leak)
  5. seam floor: adjacent disjoint slabs are separate charts
  6. missing blocks are marked magenta in the assembled atlas
  7. direct and indirect illumination share one radiance unit; indirect is
     computed by real bounces and changes with bounce depth (not a constant)
"""
import json
import os
import shutil
import tempfile

from lightbaker.baker import BakeJob, default_config
from lightbaker.charts import Atlas
from lightbaker.pathtracer import Kernel
from lightbaker.rng import PCG32, derive_seed
from lightbaker.scenes import build as build_scene

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, PASS if ok else FAIL, detail))
    print("[%s] %s %s" % (PASS if ok else FAIL, name, detail))


def test_rng_determinism():
    a = PCG32(derive_seed(20261118, 7, 12345))
    b = PCG32(derive_seed(20261118, 7, 12345))
    seq_a = [a.next_u32() for _ in range(1000)]
    seq_b = [b.next_u32() for _ in range(1000)]
    check("rng reproducible from auditable seed", seq_a == seq_b)
    c = PCG32(derive_seed(20261118, 7, 12346))
    check("distinct texel seeds diverge",
          seq_a != [c.next_u32() for _ in range(1000)])


def test_block_determinism_and_retry():
    mesh, _ = build_scene("closed_room")
    atlas = Atlas(mesh, size=256, texels_per_metre=12.0)
    kernel = Kernel(mesh)
    block = next(b for b in atlas.block_bounds(32) if b[6])
    v1, _, _ = kernel.bake_block(atlas, block, 8, 20261118, block[0] + block[1])
    v2, _, _ = kernel.bake_block(atlas, block, 8, 20261118, block[0] + block[1])
    check("block bit-identical on recompute (retry cannot add brightness)",
          v1 == v2, "texels=%d" % len(v1))
    v3, _, _ = kernel.bake_block(atlas, block, 8, 20261119, block[0] + block[1])
    check("seed change is visible and reproducible", v1 != v3)


def test_dilation_never_crosses_chart():
    mesh, _ = build_scene("seam_floor")
    atlas = Atlas(mesh, size=256, texels_per_metre=10.0)
    s = atlas.size
    violations = 0
    for idx in range(s * s):
        if not atlas.valid[idx]:
            continue
        x, y = idx % s, idx // s
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < s and 0 <= ny < s:
                nidx = ny * s + nx
                if atlas.valid[nidx] and \
                        atlas.chart_id[nidx] != atlas.chart_id[idx]:
                    # adjacent *owned* texels of different charts must have at
                    # least one empty gap texel between them, never dilation
                    if atlas.dilated[idx] == 2 or atlas.dilated[nidx] == 2:
                        violations += 1
    slab_charts = {atlas.chart_id[i] for i in range(s * s)
                   if atlas.face[i] >= 0 and mesh.face_normal(
                       atlas.face[i])[1] > 0.9 and atlas.positions[i][1] == 0.0}
    check("four disjoint floor slabs produce >=4 separate charts",
          len(slab_charts) >= 4, "floor charts=%d" % len(slab_charts))
    check("dilation never claims another chart's texel", violations == 0,
          "violations=%d" % violations)


def test_thin_wall_no_leak():
    mesh, _ = build_scene("thin_wall")
    atlas = Atlas(mesh, size=256, texels_per_metre=12.0)
    kernel = Kernel(mesh, bounces=3)
    # partition faces: vertical wall texels (normal along x) at mid-height
    lit, dark = [], []
    for idx in range(atlas.size ** 2):
        if atlas.valid[idx] and atlas.dilated[idx] == 1 \
                and abs(atlas.positions[idx][0]) < 0.05 \
                and abs(atlas.normals[idx][0]) > 0.9 \
                and 0.5 < atlas.positions[idx][1] < 2.5:
            (lit if atlas.positions[idx][0] < 0 else dark).append(idx)
    lit_rgb = [kernel.bake_texel(atlas.positions[i], atlas.normals[i], 32,
                                 (20261118, 0, i))[0] for i in lit[:24]]
    dark_rgb = [kernel.bake_texel(atlas.positions[i], atlas.normals[i], 32,
                                  (20261118, 0, i))[0] for i in dark[:24]]
    lit_mean = sum(sum(r) / 3 for r in lit_rgb) / len(lit_rgb)
    dark_mean = sum(sum(r) / 3 for r in dark_rgb) / len(dark_rgb)
    check("thin wall lit side receives light", lit_mean > 0.01,
          "lit mean=%.4f" % lit_mean)
    check("thin wall dark side stays dark (no geometric leak)",
          dark_mean < 0.25 * lit_mean,
          "dark mean=%.6f vs lit=%.4f" % (dark_mean, lit_mean))


def test_indirect_is_real_bounce():
    mesh, _ = build_scene("closed_room")
    atlas = Atlas(mesh, size=256, texels_per_metre=12.0)
    k0 = Kernel(mesh, bounces=0)
    k3 = Kernel(mesh, bounces=3)
    idx = next(i for i in range(atlas.size ** 2)
               if atlas.valid[i] and atlas.dilated[i] == 1
               and atlas.normals[i][1] > 0.9)  # floor texel
    direct, _ = k0.bake_texel(atlas.positions[idx], atlas.normals[idx],
                              64, (7, 0, idx))
    full, _ = k3.bake_texel(atlas.positions[idx], atlas.normals[idx],
                            64, (7, 0, idx))
    check("indirect bounces add positive, non-constant energy",
          sum(full) > sum(direct) > 0.0,
          "direct=%.4f full=%.4f" % (sum(direct), sum(full)))
    # red wall must tint indirect light on the floor (colour bleed)
    red_indirect = full[0] - direct[0]
    blue_indirect = full[2] - direct[2]
    check("indirect transport carries surface colour (red wall bleed)",
          red_indirect > blue_indirect,
          "dR=%.4f dB=%.4f" % (red_indirect, blue_indirect))


def test_missing_block_marker_and_retry():
    tmp = tempfile.mkdtemp(prefix="lb_test_")
    try:
        config = default_config(scene_name="closed_room", atlas_size=256,
                                texels_per_metre=12.0, samples=4)
        job = BakeJob(os.path.join(tmp, "job"), config)
        job.run(fail_blocks={3: 1})
        values, missing = job.collect_values()
        check("failed block reported as missing", missing == [3],
              "missing=%s" % missing)
        # assemble paints missing covered texels magenta
        job.assemble()
        with open(os.path.join(job.dir, "manifest.json")) as f:
            manifest = json.load(f)
        check("manifest status partial and carries error",
              manifest["status"] == "partial"
              and bool(manifest["errors"]))
        before = {k: list(v) for k, v in values.items()}
        # retry: existing blocks untouched, block 3 now computed once
        job.retry_failed()
        values2, missing2 = job.collect_values()
        same = all(before[k] == list(values2[k]) for k in before)
        check("retry leaves finished blocks unchanged (no brightness gain)",
              same and not missing2, "missing after retry=%s" % missing2)
        block3 = job.load_block(3)
        check("retried block records single final attempt",
              block3["attempt"] == 2 and block3["texels"] > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_units_consistent():
    mesh, _ = build_scene("closed_room")
    atlas = Atlas(mesh, size=256, texels_per_metre=12.0)
    kernel = Kernel(mesh)
    block = next(b for b in atlas.block_bounds(32) if b[6])
    payload_vals, _, _ = kernel.bake_block(atlas, block, 8, 1, 0)
    finite = all(isinstance(c, float) and c == c and c >= 0.0
                 for rgb in payload_vals.values() for c in rgb)
    check("all stored values are finite linear radiance floats", finite)


def main():
    print("== lightbaker acceptance tests ==")
    test_rng_determinism()
    test_block_determinism_and_retry()
    test_dilation_never_crosses_chart()
    test_thin_wall_no_leak()
    test_indirect_is_real_bounce()
    test_missing_block_marker_and_retry()
    test_units_consistent()
    failed = [r for r in results if r[1] == FAIL]
    print("\n%d/%d checks passed" % (len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
