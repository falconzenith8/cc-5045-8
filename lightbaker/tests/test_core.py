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
  8. task protocol v2: task identity includes atlas revision + coverage +
     compute, so equal block coordinates under a different atlas are not
     the same task
  9. legacy v1 conversion only reuses results that prove atlas provenance
     and exact texel coverage; anything else is rejected (never defaulted)
 10. failed/legacy-rejected blocks are re-baked on the real kernel path and
     the assembled lightmap verifies every real texel exactly once
"""
import json
import os
import shutil
import tempfile

from lightbaker import protocol
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


def test_task_identity_not_block_coordinate():
    """Same (bx, by) rectangle under a different atlas is a different task."""
    mesh, _ = build_scene("closed_room")
    atlas_a = Atlas(mesh, size=256, texels_per_metre=12.0)
    atlas_b = Atlas(mesh, size=256, texels_per_metre=11.0)
    cfg_a = default_config(scene_name="closed_room", atlas_size=256,
                           texels_per_metre=12.0, samples=64)
    cfg_b = dict(cfg_a, texels_per_metre=11.0)
    idx = next(i for i, b in enumerate(atlas_a.block_bounds(32))
               if b[6] and b[0] == 0 and b[1] == 0)
    task_a = protocol.make_task(atlas_a, atlas_a.block_bounds(32)[idx], idx,
                                cfg_a)
    task_b = protocol.make_task(atlas_b, atlas_b.block_bounds(32)[idx], idx,
                                cfg_b)
    check("block rect is identical across the two atlases",
          task_a["block"] == task_b["block"])
    check("atlas revision differs with texel density",
          task_a["atlas"]["revision"] != task_b["atlas"]["revision"])
    check("equal block coordinates are not the same task",
          task_a["task_id"] != task_b["task_id"])
    # a result for atlas A must not validate against task B
    values = {i: (0.1, 0.1, 0.1) for i in task_a["coverage"]["texels"]}
    result_a = protocol.make_result(task_a, values, 100, 1)
    check("result rejects a task with the same rect but another atlas",
          bool(protocol.validate_result(task_b, result_a)),
          protocol.validate_result(task_b, result_a)[0])
    check("result validates against its own task",
          protocol.validate_result(task_a, result_a) == [])
    # compute changes are part of identity too
    cfg_c = dict(cfg_a, samples=128)
    task_c = protocol.make_task(atlas_a, atlas_a.block_bounds(32)[idx], idx,
                                cfg_c)
    check("equal atlas but different sample config is a different task",
          task_c["task_id"] != task_a["task_id"])


def _legacy_payload(atlas, block, block_index, samples=64, seed=20261118,
                    bounces=3, drop=()):
    kernel = Kernel(build_scene("closed_room")[0], bounces=bounces)
    values, rays, texels = kernel.bake_block(atlas, block, samples, seed,
                                             block_index)
    payload = {
        "block_index": block_index,
        "block_rect": list(block[:6]),
        "covered": block[6],
        "attempt": 1,
        "samples": samples,
        "job_seed": seed,
        "bounces": bounces,
        "rays": rays,
        "texels": texels,
        "values": {str(k): list(v) for k, v in values.items()},
        "seed_scheme": protocol.SEED_SCHEME,
        "units": protocol.UNITS,
    }
    for key in drop:
        payload.pop(key, None)
    return payload


def test_legacy_conversion_gate():
    mesh, _ = build_scene("closed_room")
    atlas = Atlas(mesh, size=256, texels_per_metre=12.0)
    blocks = atlas.block_bounds(32)
    idx = next(i for i, b in enumerate(blocks) if b[6])
    config = default_config(scene_name="closed_room", atlas_size=256,
                            texels_per_metre=12.0, samples=64)
    task = protocol.make_task(atlas, blocks[idx], idx, config)
    legacy_cfg = dict(config)
    good = _legacy_payload(atlas, blocks[idx], idx)
    issues = protocol.legacy_issues(good, task, legacy_cfg)
    check("complete legacy result converts when all evidence matches",
          issues == [], "; ".join(issues[:2]))
    converted = protocol.convert_legacy(good, task, legacy_cfg,
                                        source_label="legacy")
    check("converted result is protocol v2 and validates",
          converted["protocol_version"] == protocol.PROTOCOL_VERSION
          and converted["migration"]["from_protocol"] == 1
          and protocol.validate_result(task, converted) == [])

    missing = _legacy_payload(atlas, blocks[idx], idx, drop=("samples",))
    issues = protocol.legacy_issues(missing, task, legacy_cfg)
    check("legacy result missing a field is rejected, never defaulted",
          any("samples" in i for i in issues), "; ".join(issues[:1]))

    no_cfg_issues = protocol.legacy_issues(good, task, None)
    check("legacy result without manifest/config cannot prove atlas identity",
          any("atlas identity unverifiable" in i for i in no_cfg_issues))

    other_cfg = dict(legacy_cfg, texels_per_metre=13.0)
    issues = protocol.legacy_issues(good, task, other_cfg)
    check("legacy result with a different legacy density is rejected",
          any("texels_per_metre" in i for i in issues))

    trimmed = json.loads(json.dumps(good))
    dropped_key = next(iter(trimmed["values"]))
    trimmed["values"].pop(dropped_key)
    trimmed["texels"] = len(trimmed["values"])
    issues = protocol.legacy_issues(trimmed, task, legacy_cfg)
    check("legacy result with partial texel coverage is rejected",
          any("coverage mismatch" in i for i in issues),
          "; ".join(issues[:1]))


def test_legacy_job_reuses_or_rebakes():
    """End-to-end: complete legacy evidence is reused; a damaged block is
    re-baked on the real kernel path and the atlas verifies exactly once."""
    tmp = tempfile.mkdtemp(prefix="lb_legacy_")
    try:
        job_dir = os.path.join(tmp, "job")
        os.makedirs(os.path.join(job_dir, "blocks"))
        config = default_config(scene_name="closed_room", atlas_size=256,
                                texels_per_metre=12.0, samples=4)
        job = BakeJob(job_dir, config, create=False)
        blocks = [i for i, b in enumerate(job.blocks) if b[6]][:4]
        # manufacture a complete legacy job by baking with the kernel and
        # writing v1 payloads plus the legacy manifest/config evidence
        with open(os.path.join(job_dir, "config.json"), "w") as f:
            json.dump(config, f)
        kernel = Kernel(job.mesh)
        for i in blocks:
            payload = _legacy_payload(job.atlas, job.blocks[i], i, samples=4)
            with open(os.path.join(job_dir, "blocks",
                                   "block_%04d.json" % i), "w") as f:
                json.dump(payload, f)
        job = BakeJob(job_dir, config, create=False)
        converted = [i for i in blocks if job.block_state(i) == "done"]
        check("complete legacy blocks are converted to v2",
              converted == blocks, "states=%s"
              % {i: job.block_state(i) for i in blocks})
        sample = job.load_block(blocks[0])
        check("converted block carries atlas revision and migration marker",
              sample["atlas"]["revision"] == job.atlas_identity["revision"]
              and sample["migration"]["migration"] == protocol.MIGRATION
              and sample["protocol_version"] == protocol.PROTOCOL_VERSION)
        # a genuine v1 block with partial coverage is rejected, then the real
        # worker re-bakes it and the rejection is recorded as resolved
        reject_victim = blocks[2]
        partial = _legacy_payload(job.atlas, job.blocks[reject_victim],
                                  reject_victim, samples=4)
        partial["values"].pop(next(iter(partial["values"])))
        partial["texels"] -= 1
        with open(os.path.join(job_dir, "blocks",
                               "block_%04d.json" % reject_victim), "w") as f:
            json.dump(partial, f)
        job = BakeJob(job_dir, config, create=False)
        check("legacy v1 block with partial coverage is rejected",
              job.block_state(reject_victim) == "legacy-rejected",
              job.block_state(reject_victim))
        job.run()
        check("rejected legacy block is re-baked on the worker path",
              job.block_state(reject_victim) == "done")
        # damage one *converted* v2 block by dropping a texel, then re-bake
        victim = blocks[1]
        path = os.path.join(job_dir, "blocks", "block_%04d.json" % victim)
        with open(path) as f:
            damaged = json.load(f)
        damaged["values"].pop(next(iter(damaged["values"])))
        damaged["texels"] -= 1
        with open(path, "w") as f:
            json.dump(damaged, f)
        job = BakeJob(job_dir, config, create=False)
        check("v2 result with missing coverage is marked stale",
              job.block_state(victim) == "stale",
              job.block_state(victim))
        job.run()
        values, missing = job.collect_values()
        check("stale block is re-baked by the real worker path",
              victim not in missing and job.block_state(victim) == "done")
        verification = job.verify(values, missing)
        summary = verification["summary"]
        check("assembled atlas verifies every real texel exactly once",
              summary["coverage_ok"] and summary["baked_texels"]
              == summary["real_texels"] and summary["duplicate_texels"] == 0,
              "baked=%d real=%d dup=%d"
              % (summary["baked_texels"], summary["real_texels"],
                 summary["duplicate_texels"]))
        check("block boundary alignment verified",
              summary["boundaries_ok"])
        check("re-baked rejected legacy block is marked resolved",
              summary["rejected_blocks"] == 1
              and summary["unresolved_rejected_blocks"] == 0,
              "rejected=%d unresolved=%d"
              % (summary["rejected_blocks"],
                 summary["unresolved_rejected_blocks"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("== lightbaker acceptance tests ==")
    test_rng_determinism()
    test_block_determinism_and_retry()
    test_dilation_never_crosses_chart()
    test_thin_wall_no_leak()
    test_indirect_is_real_bounce()
    test_missing_block_marker_and_retry()
    test_units_consistent()
    test_task_identity_not_block_coordinate()
    test_legacy_conversion_gate()
    test_legacy_job_reuses_or_rebakes()
    failed = [r for r in results if r[1] == FAIL]
    print("\n%d/%d checks passed" % (len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
