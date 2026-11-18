"""Bake orchestration: job config, block tasks, workers, assembly, manifest.

A bake is a directory of independent block result files plus a manifest.
Block computation is a pure function of (scene parameters, atlas geometry,
samples, seed, block index), so a retried block writes the same numbers and
never adds energy on top of an earlier attempt. Writes are atomic
(temp file + os.replace) and the job store keeps a per-block attempt history.
"""
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import imageio
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

    Recomputes the scene and atlas inside the child process, then bakes one
    block. Deterministic: same args -> same output on any worker or retry.
    """
    (job_dir, block_index, config, fail_attempts) = args
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
    mesh, _ = build_scene(config["scene"])
    atlas = Atlas(mesh, size=config["atlas_size"],
                  texels_per_metre=config["texels_per_metre"])
    blocks = atlas.block_bounds(BLOCK_SIZE)
    block = blocks[block_index]
    kernel = Kernel(mesh, bounces=config["bounces"])
    values, rays, texels = kernel.bake_block(
        atlas, block, config["samples"], config["job_seed"], block_index)
    payload = {
        "block_index": block_index,
        "block_rect": block[:6],
        "covered": block[6],
        "attempt": attempt,
        "samples": config["samples"],
        "job_seed": config["job_seed"],
        "bounces": config["bounces"],
        "rays": rays,
        "texels": texels,
        "values": {str(k): list(v) for k, v in values.items()},
        "seed_scheme": "PCG32(blake2b(job_seed, block_index, texel_index))",
        "units": "linear radiance W/(m^2 sr)",
    }
    path = os.path.join(job_dir, "blocks", "block_%04d.json" % block_index)
    _atomic_write_json(path, payload)
    return {"block_index": block_index, "attempt": attempt, "rays": rays,
            "texels": texels, "status": "done"}


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


# --------------------------------------------------------------- job store
class BakeJob:
    def __init__(self, job_dir, config, create=True):
        self.dir = job_dir
        self.blocks_dir = os.path.join(job_dir, "blocks")
        self.config = config
        if create:
            os.makedirs(self.blocks_dir, exist_ok=True)
            self._write_manifest_state("queued", [])
        mesh, scene_info = build_scene(config["scene"])
        self.mesh = mesh
        self.scene_info = scene_info
        self.atlas = Atlas(mesh, size=config["atlas_size"],
                           texels_per_metre=config["texels_per_metre"])
        self.blocks = self.atlas.block_bounds(config["block_size"])
        if create:
            self._write_manifest_state("ready", [])

    # --------------------------------------------------------- persistence
    def manifest_path(self):
        return os.path.join(self.dir, "manifest.json")

    def _write_manifest_state(self, status, errors):
        manifest = {
            "config": self.config,
            "status": status,
            "errors": errors,
            "scene_info": getattr(self, "scene_info", None),
            "block_size": self.config["block_size"],
            "padding_texels": PADDING,
            "dilation_texels": DILATION_STEPS,
        }
        if hasattr(self, "blocks"):
            manifest["blocks"] = [
                {"index": i, "rect": list(b[:6]), "covered": b[6],
                 "state": self.block_state(i)}
                for i, b in enumerate(self.blocks)]
        _atomic_write_json(self.manifest_path(), manifest)

    def block_state(self, block_index):
        if os.path.exists(os.path.join(
                self.blocks_dir, "block_%04d.json" % block_index)):
            return "done"
        return "pending"

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
        """Execute all pending blocks in a process pool.

        fail_blocks: {block_index: number_of_attempts_to_fail} demonstrates
        worker retries and kernel->panel failure feedback.
        """
        fail_blocks = dict(fail_blocks or {})
        self._write_manifest_state("running", [])
        pending = self.pending_blocks()
        errors = []
        max_workers = max(1, min(self.config["workers"], len(pending) or 1))
        if not pending:
            self._write_manifest_state("done", [])
            return self.assemble()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for bi in pending:
                args = (self.dir, bi, self.config,
                        fail_blocks.get(bi, 0))
                futures[pool.submit(bake_block_worker, args)] = bi
            for fut in as_completed(futures):
                bi = futures[fut]
                try:
                    result = fut.result()
                    if progress:
                        progress("done", result)
                except Exception as exc:  # surfaced to the panel, not fatal
                    err = {"block_index": bi, "error": str(exc),
                           "attempts": self.block_attempts(bi)}
                    errors.append(err)
                    if progress:
                        progress("failed", err)
        remaining = self.pending_blocks()
        status = "done" if not remaining else "partial"
        self._write_manifest_state(status, errors)
        return self.assemble()

    def retry_failed(self, progress=None, fail_blocks=None):
        """Re-run only missing blocks. Existing blocks are never recomputed,
        so retried areas cannot accumulate extra brightness."""
        return self.run(progress=progress, fail_blocks=fail_blocks)

    # ----------------------------------------------------------- assembly
    def collect_values(self):
        values = {}
        missing = []
        for i, block in enumerate(self.blocks):
            if not block[6]:
                continue
            if self.block_state(i) != "done":
                missing.append(i)
                continue
            payload = self.load_block(i)
            for k, rgb in payload["values"].items():
                values[int(k)] = tuple(rgb)
        return values, missing

    def assemble(self, draw_blocks=True):
        values, missing = self.collect_values()
        atlas_png = os.path.join(self.dir, "lightmap.png")
        imageio.render_atlas_png(atlas_png, self.atlas, values,
                                 missing=missing,
                                 block_size=self.config["block_size"],
                                 draw_blocks=draw_blocks)
        # display-only edge-preserving denoise; block JSON values stay raw
        denoised = imageio.edge_aware_denoise(self.atlas, values)
        imageio.render_atlas_png(
            os.path.join(self.dir, "lightmap_denoised.png"),
            self.atlas, denoised, missing=missing, draw_blocks=False)
        imageio.render_mask_png(os.path.join(self.dir, "coverage_mask.png"),
                                self.atlas)
        imageio.render_chart_id_png(os.path.join(self.dir, "chart_ids.png"),
                                    self.atlas)
        # plain block view without overlay lines
        imageio.render_atlas_png(os.path.join(self.dir, "lightmap_clean.png"),
                                 self.atlas, values, missing=missing,
                                 draw_blocks=False)
        runtime = self.atlas.runtime_mesh()
        _atomic_write_json(os.path.join(self.dir, "runtime_mesh.json"),
                           runtime)
        metrics = self.metrics(values, missing)
        _atomic_write_json(os.path.join(self.dir, "metrics.json"), metrics)
        self._write_manifest_state(
            "done" if not missing else "partial", self._read_errors())
        return metrics

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
            "atlas_size": self.config["atlas_size"],
            "samples_per_texel": self.config["samples"],
            "job_seed": self.config["job_seed"],
            "bounces": self.config["bounces"],
            "texels_covered": covered,
            "texels_real_coverage": real,
            "texels_dilated": covered - real,
            "texels_baked": len(values),
            "blocks_total": sum(1 for b in self.blocks if b[6]),
            "blocks_missing": missing,
            "mean_radiance": mean,
            "total_radiance_energy": energy,
            "units": "linear radiance W/(m^2 sr)",
            "indirect": "explicit Lambertian path bounces (depth>=1), "
                        "not an ambient constant",
            "rng": "PCG32, seed=blake2b(job_seed, block, texel, sample)",
        }


def run_bake(job_dir, config, progress=None, fail_blocks=None):
    job = BakeJob(job_dir, config)
    return job, job.run(progress=progress, fail_blocks=fail_blocks)
