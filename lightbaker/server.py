"""HTTP task panel driven entirely by the baking kernel.

The panel never invents options: allowed scenes, atlas sizes, quality presets
and block geometry are served from the kernel modules. Block failures raised
inside worker processes are captured per block and returned to the panel,
where the failed blocks can be retried without recomputing finished ones.
"""
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .baker import (BakeJob, PRESET_SAMPLES, BLOCK_SIZE, default_config)
from .charts import ATLAS_CHOICES
from .pathtracer import DEFAULT_BOUNCES
from .scenes import SCENES

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class JobRegistry:
    def __init__(self, workspace):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()

    def create(self, config):
        job_id = uuid.uuid4().hex[:12]
        job_dir = os.path.join(self.workspace, job_id)
        job = BakeJob(job_dir, config)
        with self.lock:
            self.jobs[job_id] = {"job": job, "thread": None,
                                 "progress": [], "config": config}
        return job_id, job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self):
        with self.lock:
            return sorted(self.jobs.keys(), reverse=True)


REGISTRY = None


def _panel_payload(entry):
    job = entry["job"]
    blocks = []
    for i, b in enumerate(job.blocks):
        state = job.block_state(i) if b[6] else "empty"
        attempts = job.block_attempts(i)
        blocks.append({"index": i, "rect": list(b[:6]), "covered": b[6],
                       "state": state, "attempts": attempts})
    errors = job._read_errors()
    status = job._read_status()
    return {
        "job_id": os.path.basename(job.dir),
        "status": status,
        "config": entry["config"],
        "blocks": blocks,
        "block_size": BLOCK_SIZE,
        "errors": errors,
        "atlas_size": job.atlas.size,
        "chart_count": len(job.atlas.charts),
        "progress_events": entry["progress"][-40:],
        "outputs": {
            "lightmap": "/api/jobs/%s/file/lightmap.png" % os.path.basename(job.dir),
            "lightmap_clean": "/api/jobs/%s/file/lightmap_clean.png"
                              % os.path.basename(job.dir),
            "lightmap_denoised": "/api/jobs/%s/file/lightmap_denoised.png"
                                 % os.path.basename(job.dir),
            "coverage_mask": "/api/jobs/%s/file/coverage_mask.png"
                             % os.path.basename(job.dir),
            "chart_ids": "/api/jobs/%s/file/chart_ids.png"
                         % os.path.basename(job.dir),
            "runtime_mesh": "/api/jobs/%s/file/runtime_mesh.json"
                            % os.path.basename(job.dir),
            "metrics": "/api/jobs/%s/file/metrics.json"
                       % os.path.basename(job.dir),
        },
    }


def _run_job(job_id, fail_blocks):
    entry = REGISTRY.get(job_id)
    job = entry["job"]

    def progress(kind, payload):
        entry["progress"].append({"kind": kind, **payload})

    try:
        job.run(progress=progress, fail_blocks=fail_blocks)
    except Exception as exc:  # kernel failure feeds back to the panel
        entry["progress"].append({"kind": "fatal", "error": str(exc)})


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "LightbakerPanel/1.0"

    def log_message(self, fmt, *args):
        pass

    # ------------------------------------------------------------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._file(os.path.join(WEB_DIR, "index.html"),
                              "text/html; charset=utf-8")
        if path == "/app.js":
            return self._file(os.path.join(WEB_DIR, "app.js"),
                              "application/javascript")
        if path == "/style.css":
            return self._file(os.path.join(WEB_DIR, "style.css"),
                              "text/css")
        if path == "/api/capabilities":
            return self._json({
                "scenes": {name: fn()[1] for name, fn in SCENES.items()},
                "atlas_choices": list(ATLAS_CHOICES),
                "quality_presets": PRESET_SAMPLES,
                "block_size": BLOCK_SIZE,
                "default_bounces": DEFAULT_BOUNCES,
                "kernel": "unidirectional path tracer, NEE area lights + "
                          "Lambertian bounce indirect, PCG32 deterministic",
            })
        if path == "/api/jobs":
            return self._json({"jobs": REGISTRY.list_jobs()})
        prefix = "/api/jobs/"
        if path.startswith(prefix):
            rest = path[len(prefix):].split("/")
            job_id = rest[0]
            entry = REGISTRY.get(job_id)
            if entry is None:
                return self._json({"error": "unknown job"}, 404)
            if len(rest) == 1:
                return self._json(_panel_payload(entry))
            if len(rest) == 3 and rest[1] == "file":
                name = rest[2]
                allowed = {"lightmap.png", "lightmap_clean.png",
                           "lightmap_denoised.png",
                           "coverage_mask.png", "chart_ids.png",
                           "runtime_mesh.json", "metrics.json",
                           "manifest.json"}
                if name not in allowed:
                    return self._json({"error": "forbidden file"}, 403)
                ctype = ("image/png" if name.endswith(".png")
                         else "application/json")
                return self._file(
                    os.path.join(entry["job"].dir, name), ctype)
        return self._json({"error": "not found"}, 404)

    # --------------------------------------------------------------- POST
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON body"}, 400)
        if path == "/api/jobs":
            try:
                config = default_config(
                    scene_name=body.get("scene", "closed_room"),
                    atlas_size=int(body.get("atlas_size", 256)),
                    texels_per_metre=float(body.get("density", 12.0)),
                    samples=int(body.get("samples",
                                         PRESET_SAMPLES.get(
                                             body.get("quality", "medium"),
                                             16))),
                    job_seed=int(body.get("job_seed", 20261118)),
                    bounces=int(body.get("bounces", DEFAULT_BOUNCES)),
                    workers=int(body.get("workers", 0) or
                                (os.cpu_count() or 2)))
                if body.get("quality") and not body.get("samples"):
                    config["samples"] = PRESET_SAMPLES[body["quality"]]
                job_id, job = REGISTRY.create(config)
            except Exception as exc:  # e.g. atlas too small -> panel error
                return self._json({"error": str(exc)}, 400)
            fail_blocks = {int(k): int(v)
                           for k, v in (body.get("fail_blocks") or {}).items()}
            thread = threading.Thread(target=_run_job,
                                      args=(job_id, fail_blocks), daemon=True)
            REGISTRY.jobs[job_id]["thread"] = thread
            thread.start()
            return self._json({"job_id": job_id,
                               "status": _panel_payload(
                                   REGISTRY.get(job_id))}, 201)
        # retry only missing blocks
        if path.endswith("/retry"):
            job_id = path.split("/")[3]
            entry = REGISTRY.get(job_id)
            if entry is None:
                return self._json({"error": "unknown job"}, 404)
            job = entry["job"]
            fail_blocks = {int(k): int(v)
                           for k, v in (body.get("fail_blocks") or {}).items()}
            thread = threading.Thread(
                target=lambda: job.run(
                    progress=lambda k, p: entry["progress"].append(
                        {"kind": k, **p}),
                    fail_blocks=fail_blocks), daemon=True)
            thread.start()
            return self._json({"retried": True})
        return self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8000, workspace="out/jobs"):
    global REGISTRY
    REGISTRY = JobRegistry(workspace)
    httpd = ThreadingHTTPServer((host, port), PanelHandler)
    print("Lightbaker task panel: http://%s:%d" % (host, port))
    print("Job workspace: %s" % REGISTRY.workspace)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\npanel stopped")
