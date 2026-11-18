"""Bake orchestration: versioned tasks, workers, assembly and evidence.

A bake is a directory of independent block result files plus a manifest. Each
block is a task whose identity is defined by the protocol module: the atlas
revision, the exact real-texel coverage and the compute configuration. Block
coordinates alone are never an identity, so a result produced for a different
atlas revision can never be mistaken for the current one even when it occupies
the same `(bx, by)` rectangle.

Workers receive the full task document, rebuild the scene and atlas, verify
that the atlas revision and coverage still match, bake the block, validate the
result against the task and only then write it atomically. The orchestrator
revalidates every result before assembling the atlas, so a stale, partial or
mixed-protocol job is reported as `partial` instead of silently contributing
wrong energy.

Legacy v1 results (no protocol version) are reused only when the legacy
manifest/config and the payload itself prove the same atlas parameters, block
rectangle, exact texel set and compute fields. Rejections are recorded with
their reasons; nothing is filled in from defaults.
"""
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import imageio, protocol
from .charts import Atlas, PADDING, DILATION_STEPS
from .pathtracer import Kernel, DEFAULT_BOUNCES
from .scenes import build as build_scene

BLOCK_SIZE = 32
PRESET_SAMPLES = {"preview": 4, "medium": 16, "high": 64, "reference": 128}
MAGENTA_MARKER = "magenta_in_atlas"


def default_config(scene_name="closed_room", atlas_size=256,
                   texels_per_metre=40.0, samples=16, job_seed=20261118,
                   bounces=DEFAULT_BOUNCES, workers=None, quality=None):
    if quality:
        samples = PRESET_SAMPLES[quality]
    return {
        "scene": scene_name,
        "atlas_size": atlas_size,
        "texels_per_metre": float(texels_per_metre),
        "samples": int(samples),
        "job_seed": int(job_seed),
        "bounces": int(bounces),
        "workers": int(workers or os.cpu_count() or 2),
        "block_size": BLOCK_SIZE,
    }


# ---------------------------------------------------------------- worker
def bake_block_worker(args):
    """Top-level worker entry (picklable for ProcessPoolExecutor).

    The worker never trusts the block index alone: it rebuilds the atlas from
    the task's own scene/atlas parameters, checks the revision and the exact
    texel coverage, bakes, and validates the result before writing.
    """
    job_dir, task, fail_attempts = args
    issues = protocol.validate_task(task)
    if issues:
        raise protocol.ProtocolError(
            "worker rejected task: " + "; ".join(issues))

    block_index = int(task["block"]["index"])
    attempt_file = os.path.join(job_dir, "blocks",
                                "block_%04d.attempt" % block_index)
    attempt = 0
    if os.path.exists(attempt_file):
        with open(attempt_file) as f:
            attempt = int(f.read().strip() or "0")
    attempt += 1
    _atomic_write_text(attempt_file, str(attempt))
    if attempt <= int(fail_attempts):
        raise RuntimeError(
            "injected failure for block %d (attempt %d of %d injected)"
            % (block_index, attempt, fail_attempts))

    # receiver and worker share one atlas identity check
    mesh, _ = build_scene(task["scene"])
    atlas = Atlas(mesh, size=task["atlas"]["size"],
                  texels_per_metre=task["atlas"]["texels_per_metre"])
    rebuilt = protocol.atlas_descriptor(atlas, {
        "scene": task["scene"],
        "atlas_size": task["atlas"]["size"],
        "texels_per_metre": task["atlas"]["texels_per_metre"],
        "block_size": task["compute"]["block_size"],
    })
    if rebuilt["revision"] != task["atlas"]["revision"]:
        raise protocol.ProtocolError(
            "atlas revision drift for block %d: task %s, rebuilt %s"
            % (block_index, task["atlas"]["revision"], rebuilt["revision"]))

    blocks = atlas.block_bounds(task["compute"]["block_size"])
    if block_index >= len(blocks):
        raise protocol.ProtocolError(
            "block %d outside atlas grid of %d blocks"
            % (block_index, len(blocks)))
    block = blocks[block_index]
    if [int(v) for v in block[:6]] != task["block"]["rect"]:
        raise protocol.ProtocolError(
            "block rect drift for block %d: task %r, atlas %r"
            % (block_index, task["block"]["rect"],
               [int(v) for v in block[:6]]))
    indices = protocol.coverage_indices(atlas, block)
    if indices != task["coverage"]["texels"]:
        raise protocol.ProtocolError(
            "coverage drift for block %d: task %d texels, atlas %d texels"
            % (block_index, len(task["coverage"]["texels"]), len(indices)))

    kernel = Kernel(mesh, bounces=task["compute"]["bounces"])
    values, rays, texels = kernel.bake_block(
        atlas, block, task["compute"]["samples"],
        task["compute"]["job_seed"], block_index)
    if texels != task["coverage"]["texel_count"]:
        raise protocol.ProtocolError(
            "kernel baked %d texels but task covers %d"
            % (texels, task["coverage"]["texel_count"]))

    result = protocol.make_result(task, values, rays, attempt)
    issues = protocol.validate_result(task, result)
    if issues:
        raise protocol.ProtocolError(
            "worker produced an invalid result: " + "; ".join(issues))
    path = os.path.join(job_dir, "blocks", "block_%04d.json" % block_index)
    _atomic_write_json(path, result)
    return {"block_index": block_index, "attempt": attempt, "rays": rays,
            "texels": texels, "task_id": task["task_id"], "status": "done"}


def _atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- job store
class BakeJob:
    def __init__(self, job_dir, config, create=True, lock_path=None):
        self.dir = job_dir
        self.blocks_dir = os.path.join(job_dir, "blocks")
        self.config = config
        self.lock = protocol.load_build_lock(lock_path)
        self.protocol_version = protocol.PROTOCOL_VERSION
        mesh, scene_info = build_scene(config["scene"])
        self.mesh = mesh
        self.scene_info = scene_info
        self.atlas = Atlas(mesh, size=config["atlas_size"],
                           texels_per_metre=config["texels_per_metre"])
        self.blocks = self.atlas.block_bounds(config["block_size"])
        self.atlas_identity = protocol.atlas_descriptor(self.atlas, config)
        self.compute = protocol.compute_descriptor(config)
        self.tasks = [
            protocol.make_task(self.atlas, block, i, config,
                               atlas_desc=self.atlas_identity,
                               compute=self.compute)
            for i, block in enumerate(self.blocks)
        ]
        self._states = {}
        self._payloads = {}
        self._issues = {}
        self.migration = {"from_protocol": 1,
                          "to_protocol": protocol.PROTOCOL_VERSION,
                          "migration": protocol.MIGRATION,
                          "converted": [], "rejected": [],
                          "rejected_unresolved": []}
        self._load_migration_history()
        self._legacy_config = self._read_legacy_config()
        if create:
            os.makedirs(self.blocks_dir, exist_ok=True)
        self._reconcile()
        if create:
            self._write_manifest_state("queued", [])
            self._write_manifest_state("ready", [])

    # ----------------------------------------------------- protocol metadata
    def task(self, block_index):
        return self.tasks[block_index]

    def _load_migration_history(self):
        """Reload the persisted migration report so history survives restarts.

        The report is keyed by the atlas revision: if the job is reopened with
        a different atlas, old entries are dropped rather than misattributed.
        """
        path = os.path.join(self.dir, "migration_report.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                report = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(report, dict):
            return
        if report.get("atlas_revision") != self.atlas_identity["revision"]:
            return
        if report.get("migration") != protocol.MIGRATION:
            return
        for key in ("converted", "rejected", "rejected_unresolved"):
            if isinstance(report.get(key), list):
                self.migration[key] = report[key]

    def _read_legacy_config(self):
        for name in ("manifest.json", "config.json"):
            path = os.path.join(self.dir, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            cfg = doc.get("config") if name == "manifest.json" else doc
            if isinstance(cfg, dict):
                return dict(cfg)
        return None

    # --------------------------------------------------------- persistence
    def manifest_path(self):
        return os.path.join(self.dir, "manifest.json")

    def _manifest_doc(self, status, errors):
        return {
            "protocol": protocol.TASK_PROTOCOL,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "result_schema": protocol.RESULT_SCHEMA,
            "kernel_revision": protocol.KERNEL_REVISION,
            "build_lock": {
                "lock_version": self.lock.get("lock_version"),
                "kernel_revision": self.lock.get("kernel_revision"),
                "protocol_version": self.lock.get("protocol_version"),
            },
            "config": self.config,
            "status": status,
            "errors": errors,
            "scene_info": self.scene_info,
            "block_size": self.config["block_size"],
            "padding_texels": PADDING,
            "dilation_texels": DILATION_STEPS,
            "atlas": self.atlas_identity,
            "compute": self.compute,
            "blocks": [
                {"index": i, "rect": list(b[:6]), "covered": b[6],
                 "state": self.block_state(i),
                 "task_id": self.tasks[i]["task_id"],
                 "coverage": {
                     "texel_count": self.tasks[i]["coverage"]["texel_count"],
                     "texel_digest": self.tasks[i]["coverage"]["texel_digest"],
                 }}
                for i, b in enumerate(self.blocks)],
            "migration": self.migration,
        }

    def _write_manifest_state(self, status, errors):
        _atomic_write_json(self.manifest_path(),
                           self._manifest_doc(status, errors))

    def _write_migration_report(self):
        _atomic_write_json(os.path.join(self.dir, "migration_report.json"),
                           {
                               "protocol": protocol.TASK_PROTOCOL,
                               "protocol_version": protocol.PROTOCOL_VERSION,
                               "migration": protocol.MIGRATION,
                               "legacy_config": self._legacy_config,
                               "atlas_revision": self.atlas_identity["revision"],
                               **self.migration,
                           })

    def _reconcile(self):
        """Classify every covered block against the current task identity.

        Migration history is accumulated, not recomputed: a block converted
        to v2 stays recorded across later reconciles, and its archive under
        ``blocks/legacy/`` is the original v1 payload.
        """
        self._states = {}
        self._payloads = {}
        self._issues = {}
        converted = {item["block_index"]: item
                     for item in self.migration.get("converted", [])}
        rejected = {item["block_index"]: item
                    for item in self.migration.get("rejected", [])}
        legacy_dir = os.path.join(self.blocks_dir, "legacy")
        for i, block in enumerate(self.blocks):
            if not block[6]:
                self._states[i] = "empty"
                continue
            path = os.path.join(self.blocks_dir, "block_%04d.json" % i)
            if not os.path.exists(path):
                self._states[i] = "pending"
                continue
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (OSError, ValueError) as exc:
                self._states[i] = "invalid"
                self._issues[i] = ["unreadable result: %s" % exc]
                continue
            task = self.tasks[i]
            if protocol.is_v2_payload(payload):
                issues = protocol.validate_result(task, payload)
                if issues:
                    self._states[i] = "stale"
                    self._issues[i] = issues
                else:
                    # a previously converted block keeps its provenance; a
                    # re-baked block is a raw v2 result with no legacy source
                    migration = payload.get("migration")
                    if migration:
                        if i not in converted:
                            converted[i] = {
                                "block_index": i,
                                "task_id": task["task_id"],
                                "atlas_revision": task["atlas"]["revision"],
                                "source_sha256":
                                    migration.get("source_sha256"),
                                "migration": migration.get("migration"),
                            }
                    else:
                        converted.pop(i, None)
                        if i in rejected:
                            rejected[i]["rebaked"] = True
                    self._states[i] = "done"
                    self._payloads[i] = payload
                continue
            # legacy v1: requires complete conversion evidence
            issues = protocol.legacy_issues(payload, task, self._legacy_config,
                                            "block_%04d.json" % i)
            if issues:
                self._states[i] = "legacy-rejected"
                self._issues[i] = issues
                rejected[i] = {"block_index": i,
                               "task_id": task["task_id"],
                               "atlas_revision":
                                   self.atlas_identity["revision"],
                               "reasons": issues}
                continue
            sha = _sha256_file(path)
            os.makedirs(legacy_dir, exist_ok=True)
            archive_path = os.path.join(legacy_dir, "block_%04d.json" % i)
            if not os.path.exists(archive_path):
                with open(path, "rb") as src, open(archive_path, "wb") as dst:
                    dst.write(src.read())
            result = protocol.convert_legacy(
                payload, task, self._legacy_config,
                source_label="block_%04d.json" % i, source_sha256=sha)
            _atomic_write_json(path, result)
            self._states[i] = "done"
            self._payloads[i] = result
            rejected.pop(i, None)
            converted[i] = {"block_index": i,
                            "task_id": task["task_id"],
                            "atlas_revision":
                                self.atlas_identity["revision"],
                            "source_sha256": sha,
                            "migration": protocol.MIGRATION}
        self.migration["converted"] = [converted[k]
                                       for k in sorted(converted)]
        self.migration["rejected"] = [rejected[k] for k in sorted(rejected)]
        self.migration["rejected_unresolved"] = [
            k for k, item in sorted(rejected.items())
            if not item.get("rebaked")]
        if self.migration["converted"] or self.migration["rejected"]:
            self._write_migration_report()

    def block_state(self, block_index):
        if block_index in self._states:
            return self._states[block_index]
        return "pending"

    def block_issues(self, block_index):
        return list(self._issues.get(block_index, []))

    def load_block(self, block_index):
        path = os.path.join(self.blocks_dir,
                            "block_%04d.json" % block_index)
        with open(path) as f:
            return json.load(f)

    def block_attempts(self, block_index):
        path = os.path.join(self.blocks_dir,
                            "block_%04d.attempt" % block_index)
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return int(f.read().strip() or "0")

    def pending_blocks(self):
        return [i for i, b in enumerate(self.blocks)
                if b[6] and self.block_state(i) != "done"]

    # ------------------------------------------------------------- running
    def run(self, progress=None, fail_blocks=None):
        """Bake all blocks whose result does not match the current task.

        fail_blocks: {block_index: number_of_attempts_to_fail} demonstrates
        worker retries and kernel->panel failure feedback.
        """
        fail_blocks = dict(fail_blocks or {})
        self._reconcile()
        self._write_manifest_state("running", [])
        pending = self.pending_blocks()
        errors = []
        max_workers = max(1, min(self.config["workers"], len(pending) or 1))
        if pending:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for bi in pending:
                    args = (self.dir, self.tasks[bi], fail_blocks.get(bi, 0))
                    futures[pool.submit(bake_block_worker, args)] = bi
                for fut in as_completed(futures):
                    bi = futures[fut]
                    try:
                        result = fut.result()
                        if progress:
                            progress("done", result)
                    except Exception as exc:  # surfaced to the panel
                        err = {"block_index": bi, "error": str(exc),
                               "attempts": self.block_attempts(bi),
                               "task_id": self.tasks[bi]["task_id"],
                               "atlas_revision":
                                   self.atlas_identity["revision"]}
                        errors.append(err)
                        if progress:
                            progress("failed", err)
        self._reconcile()
        remaining = self.pending_blocks()
        status = "done" if not remaining else "partial"
        self._write_manifest_state(status, errors)
        return self.assemble(errors)

    def retry_failed(self, progress=None, fail_blocks=None):
        """Re-run only blocks whose results do not match the task identity.
        Matching blocks are never recomputed, so retries cannot add energy."""
        return self.run(progress=progress, fail_blocks=fail_blocks)

    # ----------------------------------------------------------- assembly
    def collect_values(self):
        values = {}
        missing = []
        evidence = {}
        for i, block in enumerate(self.blocks):
            if not block[6]:
                continue
            state = self.block_state(i)
            if state != "done" or i not in self._payloads:
                missing.append(i)
                evidence[i] = {"state": state,
                               "issues": self.block_issues(i),
                               "task_id": self.tasks[i]["task_id"]}
                continue
            payload = self._payloads[i]
            task = self.tasks[i]
            expected = task["coverage"]["texels"]
            got = set(int(k) for k in payload["values"])
            if got != set(expected):
                missing.append(i)
                evidence[i] = {
                    "state": "coverage-mismatch",
                    "issues": ["result texels disagree with task coverage"],
                    "task_id": task["task_id"]}
                continue
            for key in expected:
                values[key] = tuple(payload["values"][str(key)])
            evidence[i] = {
                "state": "done",
                "task_id": task["task_id"],
                "atlas_revision": payload["atlas"]["revision"],
                "protocol_version": payload.get("protocol_version"),
                "result_schema": payload.get("result_schema"),
                "coverage_expected": len(expected),
                "coverage_baked": len(got),
                "samples_per_texel": payload["coverage"]["samples_per_texel"],
                "effective_samples_per_texel":
                    payload["coverage"]["effective_samples_per_texel"],
                "rays": payload.get("rays"),
                "attempt": payload.get("attempt"),
                "migration": payload.get("migration"),
            }
        self._evidence = evidence
        return values, missing

    def assemble(self, errors=None):
        if errors is None:
            errors = self._read_errors()
        values, missing = self.collect_values()
        atlas_png = os.path.join(self.dir, "lightmap.png")
        imageio.render_atlas_png(atlas_png, self.atlas, values,
                                 missing=missing,
                                 block_size=self.config["block_size"],
                                 draw_blocks=True)
        # display-only edge-preserving denoise; block JSON values stay raw
        denoised = imageio.edge_aware_denoise(self.atlas, values)
        imageio.render_atlas_png(
            os.path.join(self.dir, "lightmap_denoised.png"),
            self.atlas, denoised, missing=missing, draw_blocks=False)
        imageio.render_mask_png(os.path.join(self.dir, "coverage_mask.png"),
                                self.atlas)
        imageio.render_chart_id_png(os.path.join(self.dir, "chart_ids.png"),
                                    self.atlas)
        imageio.render_atlas_png(os.path.join(self.dir, "lightmap_clean.png"),
                                 self.atlas, values, missing=missing,
                                 draw_blocks=False)
        runtime = self.atlas.runtime_mesh()
        _atomic_write_json(os.path.join(self.dir, "runtime_mesh.json"),
                           runtime)
        verification = self.verify(values, missing)
        _atomic_write_json(os.path.join(self.dir, "verification.json"),
                           verification)
        metrics = self.metrics(values, missing)
        metrics["verification"] = verification["summary"]
        _atomic_write_json(os.path.join(self.dir, "metrics.json"), metrics)
        self._write_migration_report()
        self._write_manifest_state(
            "done" if not missing else "partial", errors or [])
        return metrics

    def verify(self, values, missing):
        """Cross-check the assembled lightmap against block boundaries.

        Every real texel must be baked exactly once, block rectangles must lie
        on the atlas grid and match the task rect, and no texel may be shared
        between two blocks. Missing/partial blocks are reported explicitly.
        """
        s = self.atlas.size
        block_size = self.config["block_size"]
        real_texels = set(i for i in range(s * s)
                          if self.atlas.valid[i]
                          and self.atlas.dilated[i] == 1)
        seen = {}
        duplicates = []
        blocks = []
        for i, block in enumerate(self.blocks):
            if not block[6]:
                continue
            bx, by, x0, y0, x1, y1, _ = block
            task = self.tasks[i]
            entry = {
                "block_index": i,
                "rect": [x0, y0, x1, y1],
                "task_id": task["task_id"],
                "state": self.block_state(i),
                "expected_texels": task["coverage"]["texel_count"],
                "baked_texels": 0,
                "atlas_revision": self.atlas_identity["revision"],
                "boundary_aligned": (x0 % block_size == 0
                                     and y0 % block_size == 0
                                     and (x1 % block_size == 0
                                          or x1 == s)
                                     and (y1 % block_size == 0
                                          or y1 == s)),
                "rect_matches_task": ([x0, y0, x1, y1]
                                      == [task["block"]["rect"][2],
                                          task["block"]["rect"][3],
                                          task["block"]["rect"][4],
                                          task["block"]["rect"][5]]),
            }
            if i in self._payloads:
                for key in self._payloads[i]["values"]:
                    idx = int(key)
                    seen[idx] = seen.get(idx, 0) + 1
                    if seen[idx] == 2:
                        duplicates.append(idx)
                    entry["baked_texels"] += 1
            blocks.append(entry)
        uncovered = sorted(real_texels - set(values))
        unknown = sorted(set(values) - real_texels)
        duplicate_ok = not duplicates
        boundaries_ok = all(b["boundary_aligned"] and b["rect_matches_task"]
                            for b in blocks)
        coverage_ok = (not uncovered and not unknown and duplicate_ok
                       and not missing)
        summary = {
            "atlas_revision": self.atlas_identity["revision"],
            "protocol_version": protocol.PROTOCOL_VERSION,
            "migration": protocol.MIGRATION,
            "real_texels": len(real_texels),
            "baked_texels": len(values),
            "uncovered_texels": len(uncovered),
            "unknown_texels": len(unknown),
            "duplicate_texels": len(duplicates),
            "missing_blocks": list(missing),
            "boundaries_ok": boundaries_ok,
            "coverage_ok": coverage_ok,
            "blocks": len(blocks),
            "converted_blocks": len(self.migration["converted"]),
            "rejected_blocks": len(self.migration["rejected"]),
            "unresolved_rejected_blocks":
                len(self.migration["rejected_unresolved"]),
        }
        return {
            "protocol": protocol.TASK_PROTOCOL,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "migration": protocol.MIGRATION,
            "atlas": self.atlas_identity,
            "compute": self.compute,
            "summary": summary,
            "uncovered_texels": uncovered[:200],
            "unknown_texels": unknown[:200],
            "duplicate_texels": duplicates[:200],
            "blocks": blocks,
            "migration_report": self.migration,
        }

    def _read_errors(self):
        try:
            with open(self.manifest_path()) as f:
                return json.load(f).get("errors", [])
        except (OSError, ValueError):
            return []

    def _read_status(self):
        try:
            with open(self.manifest_path()) as f:
                return json.load(f).get("status", "unknown")
        except (OSError, ValueError):
            return "unknown"

    def metrics(self, values, missing):
        radiance = [rgb for rgb in values.values()]
        n = len(radiance) or 1
        mean = [sum(rgb[k] for rgb in radiance) / n for k in range(3)]
        energy = sum(sum(rgb) for rgb in radiance)
        covered = sum(1 for i in range(self.atlas.size ** 2)
                      if self.atlas.valid[i])
        real = sum(1 for i in range(self.atlas.size ** 2)
                   if self.atlas.valid[i] and self.atlas.dilated[i] == 1)
        return {
            "scene": self.config["scene"],
            "protocol": protocol.TASK_PROTOCOL,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "result_schema": protocol.RESULT_SCHEMA,
            "migration": protocol.MIGRATION,
            "kernel_revision": protocol.KERNEL_REVISION,
            "atlas_revision": self.atlas_identity["revision"],
            "atlas_size": self.config["atlas_size"],
            "samples_per_texel": self.config["samples"],
            "effective_samples_per_texel":
                self.compute["effective_samples"],
            "job_seed": self.config["job_seed"],
            "bounces": self.config["bounces"],
            "texels_covered": covered,
            "texels_real_coverage": real,
            "texels_dilated": covered - real,
            "texels_baked": len(values),
            "blocks_total": sum(1 for b in self.blocks if b[6]),
            "blocks_missing": missing,
            "tasks_total": sum(1 for b in self.blocks if b[6]),
            "legacy_converted": len(self.migration["converted"]),
            "legacy_rejected": len(self.migration["rejected"]),
            "legacy_rejected_unresolved":
                len(self.migration["rejected_unresolved"]),
            "mean_radiance": mean,
            "total_radiance_energy": energy,
            "units": protocol.UNITS,
            "indirect": "explicit Lambertian path bounces (depth>=1), "
                        "not an ambient constant",
            "rng": protocol.SEED_SCHEME,
        }


def execution_evidence(job):
    """Atlas identity and coverage attached to every block's execution proof."""
    payloads = getattr(job, "_payloads", None)
    if payloads is None:
        job.collect_values()
        payloads = job._payloads
    evidence = getattr(job, "_evidence", None)
    if evidence is None:
        job.collect_values()
        evidence = job._evidence
    return {
        "protocol": protocol.TASK_PROTOCOL,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "migration": protocol.MIGRATION,
        "atlas": job.atlas_identity,
        "compute": job.compute,
        "blocks": {str(k): v for k, v in sorted(evidence.items())},
    }


def run_bake(job_dir, config, progress=None, fail_blocks=None, lock_path=None):
    job = BakeJob(job_dir, config, lock_path=lock_path)
    return job, job.run(progress=progress, fail_blocks=fail_blocks)
