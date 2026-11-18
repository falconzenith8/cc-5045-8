"""Unidirectional path-tracing lighting kernel.

Direct illumination: next-event estimation with explicit area-light sampling.
Indirect illumination: cosine-weighted Lambertian bounces with fixed
max-depth plus Russian-roulette termination. Both share the *same* linear
radiance units (W/(m^2 sr)); tone mapping happens only at display time, so
assembling blocks never mixes energy conventions.

Every texel/sample draws from a PCG32 seeded from
(job_seed, block_id, texel_index, sample_index), so results are independent
of worker scheduling and identical on retry.
"""
import math

from . import vec
from .rng import PCG32, derive_seed

RAY_EPS = 1e-4
RR_SURVIVE = 0.8
DEFAULT_BOUNCES = 3
# bounds the NEE estimator at grazing angles (firefly suppression);
# introduces only a tiny, well-defined bias in extreme edge cases
MAX_LIGHT_WEIGHT = 8.0
# per-path radiance ceiling: removes residual multi-bounce fireflies while
# staying well above the emissive panel's own radiance (~17)
MAX_PATH_RADIANCE = 6.0


def _frame(n):
    """Orthonormal tangent frame around normal n (z = n)."""
    if abs(n[0]) > 0.9:
        t = vec.normalize((0.0, 1.0, 0.0))
    else:
        t = vec.normalize((1.0, 0.0, 0.0))
    b = vec.normalize(vec.cross(n, t))
    t = vec.cross(b, n)
    return t, b, n


def _local_to_world(local, frame):
    t, b, n = frame
    return (local[0] * t[0] + local[1] * b[0] + local[2] * n[0],
            local[0] * t[1] + local[1] * b[1] + local[2] * n[1],
            local[0] * t[2] + local[1] * b[2] + local[2] * n[2])


class Kernel:
    def __init__(self, mesh, bounces=DEFAULT_BOUNCES):
        self.mesh = mesh
        self.bounces = bounces
        self.light_faces = mesh.emissive_faces()
        self.light_areas = [mesh._area[f] for f in self.light_faces]
        total = sum(self.light_areas) or 1.0
        self.light_cdf = []
        acc = 0.0
        for a in self.light_areas:
            acc += a / total
            self.light_cdf.append(acc)
        self.total_light_area = total

    # ------------------------------------------------------- light sampling
    def _sample_light(self, rng):
        r = rng.uniform2()
        fid = 0
        for i, c in enumerate(self.light_cdf):
            if r <= c:
                fid = i
                break
        face = self.light_faces[fid]
        u, v = rng.uniform_triangle()
        point = self.mesh.face_point(face, u, v)
        normal = self.mesh._normal[face]
        mat = self.mesh.materials[self.mesh.faces[face][3]]
        prob_face = self.light_areas[fid] / self.total_light_area
        return face, point, normal, mat["emission"], prob_face

    def _unoccluded(self, p_from, p_to):
        d = vec.sub(p_to, p_from)
        dist = vec.length(d)
        if dist <= RAY_EPS:
            return True
        w = vec.mul(d, 1.0 / dist)
        origin = vec.add(p_from, vec.mul(w, RAY_EPS))
        hit = self.mesh.intersect(origin, w, dist - 2.0 * RAY_EPS)
        return hit is None

    def _direct_light(self, pos, normal, albedo, rng, n_samples=2):
        """Next-event direct illumination, averaged over n_samples area-light
        draws. Averaging independent light samples cuts direct-light variance
        (the dominant Monte-Carlo term) without changing the expected value or
        the linear radiance unit."""
        if not self.light_faces:
            return (0.0, 0.0, 0.0)
        acc = [0.0, 0.0, 0.0]
        for _ in range(n_samples):
            _, lpoint, lnormal, emission, p_face = self._sample_light(rng)
            to_light = vec.sub(lpoint, pos)
            dist2 = max(vec.dot(to_light, to_light), 1e-9)
            dist = math.sqrt(dist2)
            w = vec.mul(to_light, 1.0 / dist)
            ndl = vec.dot(normal, w)
            # two-sided area light: emit on both faces
            lnd = abs(vec.dot(lnormal, vec.mul(w, -1.0)))
            if ndl <= 0.0 or lnd <= 1e-6:
                continue
            if not self._unoccluded(pos, lpoint):
                continue
            # solid-angle pdf implied by the area sample; clamp the
            # grazing-angle 1/cos(light) singularity against fireflies
            lnd_eff = max(lnd, 0.15)
            pdf = p_face * dist2 / lnd_eff
            weight = min(MAX_LIGHT_WEIGHT, ndl / (math.pi * pdf))
            for k in range(3):
                acc[k] += emission[k] * albedo[k] * weight
        inv = 1.0 / n_samples
        return (acc[0] * inv, acc[1] * inv, acc[2] * inv)

    # ------------------------------------------------------------- radiance
    def trace(self, origin, direction, rng, depth=0):
        """Radiance along a ray. Direct light via NEE, bounces via cosine."""
        hit = self.mesh.intersect(origin, direction)
        if hit is None:
            return (0.0, 0.0, 0.0)  # closed rooms have no environment term
        t, fid, u, v = hit
        mat = self.mesh.materials[self.mesh.faces[fid][3]]
        pos = vec.add(origin, vec.mul(direction, t))
        normal = self.mesh.face_normal(fid, direction)
        albedo = mat["albedo"]
        # All emissive transport is counted exactly once through next-event
        # estimation at the preceding path vertex. A ray that lands on a
        # light therefore contributes no raw emission at any depth: returning
        # it would double-count NEE light and inject the small-area-light
        # firefly tail (the dominant source of salt-and-pepper noise).
        if any(mat["emission"]):
            return (0.0, 0.0, 0.0)
        direct = self._direct_light(pos, normal, albedo, rng)
        radiance = direct
        if depth >= self.bounces:
            return radiance
        if depth >= 1 and rng.uniform() > RR_SURVIVE:
            return radiance
        # cosine-weighted Lambertian continuation
        lx, ly, lz, _ = rng.cosine_hemisphere()
        frame = _frame(normal)
        w = _local_to_world((lx, ly, lz), frame)
        next_origin = vec.add(pos, vec.mul(normal, RAY_EPS))
        indirect = self.trace(next_origin, w, rng, depth + 1)
        rr = 1.0 if depth == 0 else (1.0 / RR_SURVIVE)
        bounced = (albedo[0] * indirect[0],
                   albedo[1] * indirect[1],
                   albedo[2] * indirect[2])
        radiance = vec.add(radiance, vec.mul(bounced, rr))
        # single-sample firefly clamp on the final path estimator. NEE keeps
        # the expected value unbiased for ordinary geometry; this only bounds
        # rare multiple-bounce paths that happen to reach the small light.
        return (min(radiance[0], MAX_PATH_RADIANCE),
                min(radiance[1], MAX_PATH_RADIANCE),
                min(radiance[2], MAX_PATH_RADIANCE))

    # ------------------------------------------------------------ texel job
    def bake_texel(self, position, normal, samples, seed_parts, face_id=None):
        # A texel sitting on an emissive surface is the light source: its
        # outgoing radiance is its own emission, independent of hemisphere
        # sampling. Everything else goes through the path tracer.
        if face_id is not None:
            mat = self.mesh.materials[self.mesh.faces[face_id][3]]
            if any(mat["emission"]):
                return tuple(mat["emission"]), 0
        rng = PCG32(derive_seed(*seed_parts), seq=0xC0FFEE)
        frame = _frame(normal)
        side = int(math.ceil(math.sqrt(max(1.0, float(samples)))))
        total = side * side
        acc = [0.0, 0.0, 0.0]
        origin = vec.add(position, vec.mul(normal, RAY_EPS))
        for s in range(total):
            # stratified jitter inside a sqrt(samples) grid
            jx = ((s % side) + rng.uniform()) / side
            jy = ((s // side) + rng.uniform()) / side
            r2 = jx
            phi = 2.0 * math.pi * jy
            r = math.sqrt(r2)
            local = (r * math.cos(phi), r * math.sin(phi),
                     math.sqrt(max(0.0, 1.0 - r2)))
            direction = _local_to_world(local, frame)
            L = self.trace(origin, direction, rng, depth=0)
            acc[0] += L[0]
            acc[1] += L[1]
            acc[2] += L[2]
        inv = 1.0 / total
        return (acc[0] * inv, acc[1] * inv, acc[2] * inv), total

    def bake_block(self, atlas, block, samples, job_seed, block_index):
        """Bake every covered texel in one atlas block.

        Returns (values, rays, texels). Values are linear radiance floats;
        the same block recomputed with the same arguments is bit-identical.
        """
        bx, by, x0, y0, x1, y1, covered = block
        if not covered:
            return {}, 0, 0
        size = atlas.size
        values = {}
        rays = 0
        texels = 0
        for y in range(y0, y1):
            row = y * size
            for x in range(x0, x1):
                idx = row + x
                if not atlas.valid[idx] or atlas.dilated[idx] == 2:
                    continue  # dilation texels are filled by the assembler
                texels += 1
                rgb, used = self.bake_texel(
                    atlas.positions[idx], atlas.normals[idx], samples,
                    (job_seed, block_index, idx), atlas.face[idx])
                values[idx] = rgb
                rays += used
        return values, rays, texels
