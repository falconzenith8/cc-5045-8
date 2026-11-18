"""Versioned task/result protocol for distributed light baking.

Protocol v2 makes the identity of a bake explicit so two results can never be
confused merely because they occupy the same atlas block coordinates.

Every task carries:
  * ``atlas``    - atlas revision (digest over chart segmentation, texel
                   positions/normals/faces/barycentrics and coverage) plus
                   the atlas parameters that produced it.
  * ``coverage`` - the exact set of real (non-dilation) texels the task owns,
                   its count and a digest, and the per-texel sample budget.
  * ``compute``  - kernel revision, samples, seed, bounce depth, block size,
                   units and seed scheme.

``task_id`` is a digest over atlas revision + block rect + coverage digest +
compute descriptor, so changing the atlas, density or sample configuration
produces a different task even when the block rectangle is identical.

Legacy v1 results (the pre-protocol block payloads) carry no atlas identity.
They may only be reused after a *complete conversion check*: the legacy
manifest/config must prove the atlas parameters, the block rect must match,
the value keys must equal the current task's exact real-texel set, and the
compute fields must match field-for-field. Anything missing or different is a
rejection, never filled in with a default.

The build lock (``bake.lock.json``) pins the protocol version and kernel
revision. Tasks/results from a different lock are not interchangeable.
"""
import hashlib
import json
import math
import os
import struct

PROTOCOL_VERSION = 2
RESULT_SCHEMA = 2
ATLAS_SCHEMA = 2
TASK_PROTOCOL = "lightbaker-bake"
MIGRATION = "v1->v2"
KERNEL_REVISION = "pathtracer-2026.11-r1"
UNITS = "linear radiance W/(m^2 sr)"
SEED_SCHEME = "PCG32(blake2b(job_seed, block_index, texel_index))"
LOCK_NAME = "bake.lock.json"

LEGACY_REQUIRED_FIELDS = (
    "block_index", "block_rect", "covered", "samples", "job_seed",
    "bounces", "texels", "rays", "values", "units", "seed_scheme",
)


class ProtocolError(RuntimeError):
    """Raised when a task/result cannot be trusted under the protocol."""


def _b2(*parts, size=16):
    h = hashlib.blake2b(digest_size=size)
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _json_digest(obj, size=16):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=size).hexdigest()


def _f(x):
    return struct.pack("<d", float(x))


def effective_samples(samples):
    side = int(math.ceil(math.sqrt(max(1.0, float(samples)))))
    return side * side


# ------------------------------------------------------------------ atlas
def coverage_indices(atlas, block):
    """Real-coverage texel indices owned by one block, row-major."""
    _, _, x0, y0, x1, y1, covered = block
    if not covered:
        return []
    size = atlas.size
    out = []
    for y in range(y0, y1):
        row = y * size
        for x in range(x0, x1):
            idx = row + x
            if atlas.valid[idx] and atlas.dilated[idx] == 1:
                out.append(idx)
    return out


def texel_digest(indices):
    h = hashlib.blake2b(digest_size=16)
    for idx in sorted(indices):
        h.update(struct.pack("<q", int(idx)))
    return h.hexdigest()


def atlas_descriptor(atlas, config):
    """Explicit atlas identity: revision digest + coverage evidence."""
    size = atlas.size
    tpm = float(atlas.tpm)
    block_size = int(config["block_size"])
    chart_h = hashlib.blake2b(digest_size=16)
    charts = sorted(atlas.charts,
                    key=lambda c: (tuple(c["bbox"]), tuple(sorted(c["faces"])),
                                   int(c["cid"])))
    for ch in charts:
        chart_h.update(struct.pack("<i", int(ch["cid"])))
        chart_h.update(struct.pack("<4i", *[int(v) for v in ch["bbox"]]))
        chart_h.update(struct.pack("<2i", int(ch["w"]), int(ch["h"])))
        chart_h.update(struct.pack("<2i", int(ch["origin"][0]),
                                   int(ch["origin"][1])))
        for fid in sorted(ch["faces"]):
            chart_h.update(struct.pack("<i", int(fid)))

    tex_h = hashlib.blake2b(digest_size=32)
    real = dilated = valid = 0
    for idx in range(size * size):
        if not atlas.valid[idx]:
            continue
        valid += 1
        if atlas.dilated[idx] == 1:
            real += 1
        elif atlas.dilated[idx] == 2:
            dilated += 1
        tex_h.update(struct.pack("<4i", idx, int(atlas.chart_id[idx]),
                                 int(atlas.face[idx]), int(atlas.dilated[idx])))
        for arr in (atlas.positions[idx], atlas.normals[idx], atlas.bary[idx]):
            if arr is None:
                tex_h.update(b"\x00" * 24)
            else:
                for value in arr:
                    tex_h.update(_f(value))

    revision = _b2(
        "lightbaker-atlas", ATLAS_SCHEMA, config["scene"], size, tpm,
        block_size, bytes.fromhex(chart_h.hexdigest()),
        bytes.fromhex(tex_h.hexdigest()), size=16)
    return {
        "schema": ATLAS_SCHEMA,
        "revision": revision,
        "scene": config["scene"],
        "size": size,
        "texels_per_metre": tpm,
        "block_size": block_size,
        "chart_count": len(atlas.charts),
        "coverage": {
            "real_texels": real,
            "dilation_texels": dilated,
            "valid_texels": valid,
        },
        "chart_digest": chart_h.hexdigest(),
        "texel_digest": tex_h.hexdigest(),
    }


def compute_descriptor(config):
    samples = int(config["samples"])
    return {
        "kernel_revision": KERNEL_REVISION,
        "samples": samples,
        "effective_samples": effective_samples(samples),
        "job_seed": int(config["job_seed"]),
        "bounces": int(config["bounces"]),
        "block_size": int(config["block_size"]),
        "units": UNITS,
        "seed_scheme": SEED_SCHEME,
    }


# -------------------------------------------------------------- task/result
def task_digest(task):
    return _json_digest({
        "protocol": task["protocol"],
        "protocol_version": task["protocol_version"],
        "scene": task["scene"],
        "atlas_revision": task["atlas"]["revision"],
        "block": task["block"],
        "coverage": {
            "texel_count": task["coverage"]["texel_count"],
            "texel_digest": task["coverage"]["texel_digest"],
        },
        "compute": task["compute"],
    })


def make_task(atlas, block, block_index, config, atlas_desc=None,
              compute=None):
    atlas_desc = atlas_desc or atlas_descriptor(atlas, config)
    compute = compute or compute_descriptor(config)
    indices = coverage_indices(atlas, block)
    task = {
        "protocol": TASK_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "scene": config["scene"],
        "atlas": atlas_desc,
        "block": {"index": int(block_index),
                  "rect": [int(v) for v in block[:6]]},
        "coverage": {
            "texel_count": len(indices),
            "texel_digest": texel_digest(indices),
            "texels": indices,
            "samples_per_texel": compute["samples"],
            "effective_samples_per_texel": compute["effective_samples"],
        },
        "compute": compute,
    }
    task["task_id"] = task_digest(task)
    return task


def result_coverage(task):
    cov = task["coverage"]
    return {
        "texel_count": cov["texel_count"],
        "texel_digest": cov["texel_digest"],
        "samples_per_texel": cov["samples_per_texel"],
        "effective_samples_per_texel": cov["effective_samples_per_texel"],
    }


def validate_task(task):
    issues = []
    if not isinstance(task, dict):
        return ["task is not an object"]
    for key in ("protocol", "protocol_version", "scene", "atlas", "block",
                "coverage", "compute", "task_id"):
        if key not in task:
            issues.append("missing task field %r" % key)
    if issues:
        return issues
    if task["protocol"] != TASK_PROTOCOL:
        issues.append("unknown task protocol %r" % (task["protocol"],))
    if task["protocol_version"] != PROTOCOL_VERSION:
        issues.append("task protocol version %r != %d"
                      % (task["protocol_version"], PROTOCOL_VERSION))
    if task["compute"].get("kernel_revision") != KERNEL_REVISION:
        issues.append("task kernel revision %r != %r"
                      % (task["compute"].get("kernel_revision"),
                         KERNEL_REVISION))
    if task["compute"].get("units") != UNITS:
        issues.append("task units mismatch")
    if task["compute"].get("seed_scheme") != SEED_SCHEME:
        issues.append("task seed scheme mismatch")
    cov = task["coverage"]
    texels = cov.get("texels")
    if not isinstance(texels, list):
        issues.append("task coverage has no explicit texel list")
    else:
        if cov.get("texel_count") != len(texels):
            issues.append("task texel_count %r != %d"
                          % (cov.get("texel_count"), len(texels)))
        if texels != sorted(set(texels)):
            issues.append("task texel list is not sorted/unique")
        if cov.get("texel_digest") != texel_digest(texels):
            issues.append("task texel digest does not match texel list")
        if cov.get("samples_per_texel") != task["compute"].get("samples"):
            issues.append("task coverage samples disagree with compute")
    if task.get("task_id") != task_digest(task):
        issues.append("task_id does not match atlas identity/coverage/"
                      "compute")
    return issues


def make_result(task, values, rays, attempt):
    vals = {}
    for key in sorted(values):
        rgb = values[key]
        vals[str(int(key))] = [float(c) for c in rgb]
    return {
        "protocol": TASK_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "result_schema": RESULT_SCHEMA,
        "task_id": task["task_id"],
        "scene": task["scene"],
        "atlas": task["atlas"],
        "block": task["block"],
        "coverage": result_coverage(task),
        "compute": task["compute"],
        "values": vals,
        "texels": len(vals),
        "rays": int(rays),
        "attempt": int(attempt),
        "units": UNITS,
        "seed_scheme": SEED_SCHEME,
    }


def _parse_value_map(raw):
    issues = []
    parsed = {}
    if not isinstance(raw, dict):
        return None, ["values is not an object"]
    for key, rgb in raw.items():
        if not isinstance(key, str):
            issues.append("value key %r is not a string" % (key,))
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            issues.append("value key %r is not an integer" % (key,))
            continue
        if str(idx) != key:
            issues.append("value key %r is not canonical" % (key,))
            continue
        if idx in parsed:
            issues.append("duplicate value key %r" % (key,))
            continue
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            issues.append("value %r is not an rgb triple" % (key,))
            continue
        try:
            chan = tuple(float(c) for c in rgb)
        except (TypeError, ValueError):
            issues.append("value %r has non-numeric channels" % (key,))
            continue
        if any((not math.isfinite(c)) or c < 0.0 for c in chan):
            issues.append("value %r has non-finite/negative channels" % (key,))
            continue
        parsed[idx] = chan
    return parsed, issues


def validate_result(task, payload):
    issues = []
    if not isinstance(payload, dict):
        return ["result is not an object"]
    for key in ("protocol", "protocol_version", "result_schema", "task_id",
                "atlas", "block", "coverage", "compute", "values", "texels",
                "rays", "units", "seed_scheme"):
        if key not in payload:
            issues.append("missing result field %r" % key)
    if issues:
        return issues
    if payload["protocol"] != TASK_PROTOCOL:
        issues.append("unknown result protocol %r" % (payload["protocol"],))
    if payload["protocol_version"] != PROTOCOL_VERSION:
        issues.append("result protocol version %r != %d"
                      % (payload["protocol_version"], PROTOCOL_VERSION))
    if payload["result_schema"] != RESULT_SCHEMA:
        issues.append("result schema %r != %d"
                      % (payload["result_schema"], RESULT_SCHEMA))
    if payload["task_id"] != task["task_id"]:
        issues.append("task_id mismatch")
    if payload["atlas"] != task["atlas"]:
        issues.append("atlas identity mismatch (revision %r != %r)"
                      % (payload["atlas"].get("revision")
                         if isinstance(payload["atlas"], dict) else None,
                         task["atlas"]["revision"]))
    if payload["block"] != task["block"]:
        issues.append("block rect/index mismatch")
    if payload["compute"] != task["compute"]:
        issues.append("compute configuration mismatch")
    if payload["coverage"] != result_coverage(task):
        issues.append("coverage evidence mismatch")
    if payload["units"] != UNITS:
        issues.append("units mismatch")
    if payload["seed_scheme"] != SEED_SCHEME:
        issues.append("seed scheme mismatch")

    parsed, value_issues = _parse_value_map(payload["values"])
    issues.extend(value_issues)
    if parsed is not None:
        expected = set(task["coverage"]["texels"])
        got = set(parsed)
        if got != expected:
            issues.append("texel coverage mismatch: %d missing, %d extra"
                          % (len(expected - got), len(got - expected)))
        if payload["texels"] != len(parsed):
            issues.append("texels field %r != %d"
                          % (payload["texels"], len(parsed)))
        count = task["coverage"]["texel_count"]
        if len(parsed) != count:
            issues.append("covered texel count %d != task %d"
                          % (len(parsed), count))
    try:
        rays = int(payload["rays"])
    except (TypeError, ValueError):
        issues.append("rays is not an integer")
        rays = -1
    if rays < 0:
        issues.append("rays is negative")
    if parsed is not None and len(parsed) == 0 and rays != 0:
        issues.append("empty result reports %d rays" % rays)
    return issues


# --------------------------------------------------------- legacy conversion
def is_v2_payload(payload):
    return isinstance(payload, dict) and "protocol_version" in payload


def legacy_issues(payload, task, legacy_config, source_label=None):
    """Return every reason a v1 payload cannot be converted to v2.

    An empty list means the result is fully attributable to this task. Missing
    fields are reported; they are never replaced by defaults.
    """
    label = source_label or "legacy block %s" % task["block"]["index"]
    issues = []
    if not isinstance(payload, dict):
        return ["%s: legacy result is not an object" % label]
    if "protocol_version" in payload:
        return ["%s: payload declares a protocol version and is not v1"
                % label]
    for key in LEGACY_REQUIRED_FIELDS:
        if key not in payload:
            issues.append("%s: missing legacy field %r" % (label, key))
    if issues:
        return issues

    if legacy_config is None:
        issues.append("%s: no legacy manifest/config, atlas identity "
                      "unverifiable" % label)
    else:
        expected = (
            ("atlas_size", task["atlas"]["size"]),
            ("texels_per_metre", task["atlas"]["texels_per_metre"]),
            ("block_size", task["compute"]["block_size"]),
            ("samples", task["compute"]["samples"]),
            ("job_seed", task["compute"]["job_seed"]),
            ("bounces", task["compute"]["bounces"]),
        )
        if "scene" not in legacy_config:
            issues.append("%s: legacy config has no 'scene', cannot prove "
                          "atlas identity" % label)
        elif legacy_config["scene"] != task["scene"]:
            issues.append("%s: legacy config scene=%r != task %r"
                          % (label, legacy_config["scene"], task["scene"]))
        for key, want in expected:
            if key not in legacy_config:
                issues.append("%s: legacy config has no %r, cannot prove "
                              "atlas/compute identity" % (label, key))
                continue
            try:
                if float(legacy_config[key]) != float(want):
                    issues.append(
                        "%s: legacy config %s=%r != task %r"
                        % (label, key, legacy_config[key], want))
            except (TypeError, ValueError):
                issues.append("%s: legacy config %s=%r is not numeric"
                              % (label, key, legacy_config[key]))

    try:
        block_index = int(payload["block_index"])
    except (TypeError, ValueError):
        block_index = None
        issues.append("%s: block_index is not an integer" % label)
    if block_index is not None and block_index != task["block"]["index"]:
        issues.append("%s: block_index %r != task %r"
                      % (label, block_index, task["block"]["index"]))

    rect = payload["block_rect"]
    if not isinstance(rect, (list, tuple)) or len(rect) != 6:
        issues.append("%s: block_rect is not six integers" % label)
    else:
        try:
            rect_ints = [int(v) for v in rect]
        except (TypeError, ValueError):
            rect_ints = None
        if rect_ints is None or rect_ints != task["block"]["rect"]:
            issues.append("%s: block rect %r != task %r"
                          % (label, rect, task["block"]["rect"]))
    if payload["covered"] is not True:
        issues.append("%s: legacy result is not marked covered" % label)

    for key, want in (("samples", task["compute"]["samples"]),
                      ("job_seed", task["compute"]["job_seed"]),
                      ("bounces", task["compute"]["bounces"])):
        try:
            got = int(payload[key])
        except (TypeError, ValueError):
            issues.append("%s: %s is not an integer" % (label, key))
            continue
        if got != int(want):
            issues.append("%s: %s=%d != task %d" % (label, key, got, want))
    if payload["units"] != UNITS:
        issues.append("%s: units %r != %r"
                      % (label, payload["units"], UNITS))
    if payload["seed_scheme"] != SEED_SCHEME:
        issues.append("%s: seed scheme %r != %r"
                      % (label, payload["seed_scheme"], SEED_SCHEME))

    parsed, value_issues = _parse_value_map(payload["values"])
    issues.extend("%s: %s" % (label, msg) for msg in value_issues)
    if parsed is not None:
        expected_texels = set(task["coverage"]["texels"])
        got_texels = set(parsed)
        if got_texels != expected_texels:
            issues.append(
                "%s: coverage mismatch, %d missing and %d extra texels"
                % (label, len(expected_texels - got_texels),
                   len(got_texels - expected_texels)))
        try:
            declared = int(payload["texels"])
        except (TypeError, ValueError):
            declared = None
            issues.append("%s: texels field is not an integer" % label)
        if declared is not None:
            if declared != len(parsed):
                issues.append("%s: texels=%r != %d value entries"
                              % (label, declared, len(parsed)))
            if declared != task["coverage"]["texel_count"]:
                issues.append("%s: texels=%r != task coverage %d"
                              % (label, declared,
                                 task["coverage"]["texel_count"]))
        try:
            rays = int(payload["rays"])
        except (TypeError, ValueError):
            rays = None
            issues.append("%s: rays is not an integer" % label)
        if rays is not None:
            if rays < 0:
                issues.append("%s: rays is negative" % label)
            if len(parsed) == 0 and rays != 0:
                issues.append("%s: empty legacy result reports %d rays"
                              % (label, rays))
    try:
        attempt = int(payload["attempt"]) if "attempt" in payload else None
    except (TypeError, ValueError):
        attempt = None
    if attempt is not None and attempt < 1:
        issues.append("%s: attempt %r < 1" % (label, attempt))
    return issues


def convert_legacy(payload, task, legacy_config, source_label=None,
                   source_sha256=None):
    issues = legacy_issues(payload, task, legacy_config, source_label)
    if issues:
        raise ProtocolError("legacy conversion rejected: " + "; ".join(issues))
    parsed, _ = _parse_value_map(payload["values"])
    attempt = int(payload["attempt"]) if "attempt" in payload else 1
    result = make_result(task, parsed, int(payload["rays"]), attempt)
    result["migration"] = {
        "from_protocol": 1,
        "to_protocol": PROTOCOL_VERSION,
        "migration": MIGRATION,
        "source": source_label,
        "source_sha256": source_sha256,
    }
    check = validate_result(task, result)
    if check:
        raise ProtocolError("converted result failed validation: "
                            + "; ".join(check))
    return result


# ---------------------------------------------------------------- build lock
def _lock_candidates(path=None):
    if path:
        yield path
    env = os.environ.get("LIGHTBAKER_BUILD_LOCK")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    yield os.path.join(root, LOCK_NAME)
    yield os.path.join(os.getcwd(), LOCK_NAME)


def load_build_lock(path=None):
    last_error = None
    for candidate in _lock_candidates(path):
        try:
            with open(candidate) as f:
                lock = json.load(f)
        except OSError as exc:
            last_error = str(exc)
            continue
        except ValueError as exc:
            raise ProtocolError("build lock %s is not valid JSON: %s"
                                % (candidate, exc))
        required = {
            "lock_version": 1,
            "task_protocol": TASK_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "result_schema": RESULT_SCHEMA,
            "atlas_schema": ATLAS_SCHEMA,
            "migration": MIGRATION,
            "kernel_revision": KERNEL_REVISION,
            "units": UNITS,
            "seed_scheme": SEED_SCHEME,
        }
        for key, want in required.items():
            if key not in lock:
                raise ProtocolError(
                    "build lock %s is missing %r; refusing to bake"
                    % (candidate, key))
            if lock[key] != want:
                raise ProtocolError(
                    "build lock %s has %s=%r but code requires %r; update "
                    "the lock and protocol version together"
                    % (candidate, key, lock[key], want))
        return lock
    raise ProtocolError("no build lock (%s) found: %s"
                        % (LOCK_NAME, last_error or "not found"))
