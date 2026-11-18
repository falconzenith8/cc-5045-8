"""Command line entry point for the light baker."""
import argparse
import json
import os
import sys
import time

from .baker import (BakeJob, PRESET_SAMPLES, default_config, run_bake,
                    BLOCK_SIZE)
from .scenes import SCENES


def _print_progress(kind, payload):
    if kind == "done":
        print("  block %3d done (attempt %d, %d texels, %d rays)"
              % (payload["block_index"], payload["attempt"],
                 payload["texels"], payload["rays"]))
    else:
        print("  block %3d FAILED: %s"
              % (payload["block_index"], payload["error"]))


def cmd_bake(args):
    config = default_config(
        scene_name=args.scene, atlas_size=args.atlas,
        texels_per_metre=args.density, job_seed=args.seed,
        bounces=args.bounces, workers=args.workers,
        quality=args.quality)
    if args.samples:
        config["samples"] = args.samples
    fail_blocks = {}
    if args.fail_block is not None:
        for spec in args.fail_block:
            bi, n = spec.split(":")
            fail_blocks[int(bi)] = int(n)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    t0 = time.time()
    job, metrics = run_bake(args.out, config, progress=_print_progress,
                            fail_blocks=fail_blocks)
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nbake finished: status=%s" % job._read_status())
    print(json.dumps(metrics, indent=2))
    print("\noutputs in %s:" % args.out)
    for name in ("lightmap.png", "lightmap_denoised.png",
                 "lightmap_clean.png", "coverage_mask.png",
                 "chart_ids.png", "runtime_mesh.json", "manifest.json"):
        p = os.path.join(args.out, name)
        print("  %-20s %s" % (name, ("ok" if os.path.exists(p) else "MISSING")))
    return 0


def cmd_scenes(args):
    print("available scenes:")
    for name, fn in sorted(SCENES.items()):
        _, info = fn()
        print("  %-12s %s" % (name, info["description"]))
    print("\nquality presets (samples/texel): %s" % PRESET_SAMPLES)
    print("atlas choices: 128, 256, 512, 1024; block size: %d texels"
          % BLOCK_SIZE)


def cmd_server(args):
    from .server import serve
    serve(args.host, args.port, args.workspace)


def cmd_demo(args):
    """Bake all three acceptance scenes at small, auditable settings."""
    os.makedirs(args.out, exist_ok=True)
    settings = {
        "closed_room": dict(atlas_size=256, texels_per_metre=12.0),
        "thin_wall": dict(atlas_size=256, texels_per_metre=12.0),
        "seam_floor": dict(atlas_size=256, texels_per_metre=10.0),
    }
    summary = {}
    for scene, st in settings.items():
        out = os.path.join(args.out, scene)
        config = default_config(scene_name=scene, samples=args.samples,
                                job_seed=args.seed, **st)
        t0 = time.time()
        _, metrics = run_bake(out, config, progress=_print_progress)
        metrics["elapsed_seconds"] = round(time.time() - t0, 2)
        summary[scene] = metrics
        print("-> %s complete in %.1fs" % (scene, metrics["elapsed_seconds"]))
    with open(os.path.join(args.out, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\ndemo outputs under %s/<scene>/" % args.out)


def build_parser():
    p = argparse.ArgumentParser(
        prog="python3 -m lightbaker.cli",
        description="Indoor showroom lightmap baker (path-traced, "
                    "atlas-based, deterministic).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bake", help="bake one scene into an atlas job folder")
    pb.add_argument("--scene", default="closed_room", choices=sorted(SCENES))
    pb.add_argument("--atlas", type=int, default=256,
                    choices=(128, 256, 512, 1024))
    pb.add_argument("--density", type=float, default=12.0,
                    help="texels per metre")
    pb.add_argument("--samples", type=int, default=0,
                    help="override samples per texel (else use --quality)")
    pb.add_argument("--quality", choices=sorted(PRESET_SAMPLES),
                    default="medium")
    pb.add_argument("--bounces", type=int, default=3)
    pb.add_argument("--seed", type=int, default=20261118)
    pb.add_argument("--workers", type=int, default=None)
    pb.add_argument("--out", required=True)
    pb.add_argument("--fail-block", action="append",
                    help="inject failures, e.g. 7:2 fails block 7 twice")
    pb.set_defaults(func=cmd_bake)

    ps = sub.add_parser("scenes", help="list scenes and presets")
    ps.set_defaults(func=cmd_scenes)

    pd = sub.add_parser("demo", help="bake all three acceptance scenes")
    pd.add_argument("--out", default="out/demo")
    pd.add_argument("--samples", type=int, default=64)
    pd.add_argument("--seed", type=int, default=20261118)
    pd.set_defaults(func=cmd_demo)

    pv = sub.add_parser("server", help="launch the bake task panel")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--workspace", default="out/jobs")
    pv.set_defaults(func=cmd_server)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
