"""Triangle meshes with a midpoint-split BVH and double-sided ray intersection.

Triangles are double-sided on purpose: an infinitely thin partition wall is a
single layer of triangles, and back-face culling would let light leak through
it. The shading normal is flipped to face the incoming ray instead.
"""
from . import vec

_EPS = 1e-7


class Mesh:
    def __init__(self, name="mesh"):
        self.name = name
        self.verts = []
        self.faces = []          # (i0, i1 i2, material_id)
        self.materials = []      # {"name", "albedo":(rgb), "emission":(rgb)}
        # cached per-face data
        self._v0 = []
        self._e1 = []
        self._e2 = []
        self._normal = []
        self._area = []
        self._centroid = []
        self._bvh = None

    def add_material(self, name, albedo=(0.8, 0.8, 0.8), emission=(0.0, 0.0, 0.0)):
        self.materials.append({"name": name, "albedo": tuple(albedo),
                               "emission": tuple(emission)})
        return len(self.materials) - 1

    def add_vertex(self, p):
        self.verts.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.verts) - 1

    def add_face(self, i0, i1, i2, material_id=0):
        v0, v1, v2 = self.verts[i0], self.verts[i1], self.verts[i2]
        e1 = vec.sub(v1, v0)
        e2 = vec.sub(v2, v0)
        n = vec.cross(e1, e2)
        area2 = vec.length(n)
        if area2 <= 1e-12:
            raise ValueError("degenerate triangle %s %s %s" % (i0, i1, i2))
        self.faces.append((i0, i1, i2, material_id))
        self._v0.append(v0)
        self._e1.append(e1)
        self._e2.append(e2)
        self._normal.append(vec.mul(n, 1.0 / area2))
        self._area.append(0.5 * area2)
        self._centroid.append(vec.mul(vec.add(vec.add(v0, v1), v2), 1.0 / 3.0))
        return len(self.faces) - 1

    def add_quad(self, a, b, c, d, material_id=0):
        i0 = self.add_vertex(a)
        i1 = self.add_vertex(b)
        i2 = self.add_vertex(c)
        i3 = self.add_vertex(d)
        self.add_face(i0, i1, i2, material_id)
        self.add_face(i0, i2, i3, material_id)

    def add_box(self, lo, hi, material_id=0):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        ids = [self.add_vertex(q) for q in p]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                 (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
        for q in quads:
            a, b, c, d = (ids[k] for k in q)
            self.add_face(a, b, c, material_id)
            self.add_face(a, c, d, material_id)

    def face_normal(self, fid, ray_dir=None):
        n = self._normal[fid]
        if ray_dir is not None and vec.dot(n, ray_dir) > 0.0:
            return (-n[0], -n[1], -n[2])
        return n

    def face_point(self, fid, u, v):
        v0 = self._v0[fid]
        return vec.add(v0, vec.add(vec.mul(self._e1[fid], u),
                                   vec.mul(self._e2[fid], v)))

    def emissive_faces(self):
        out = []
        for fid, (_, _, _, mid) in enumerate(self.faces):
            if any(self.materials[mid]["emission"]):
                out.append(fid)
        return out

    def total_emissive_area(self):
        total = 0.0
        for fid in self.emissive_faces():
            total += self._area[fid]
        return total

    def build_bvh(self):
        ids = list(range(len(self.faces)))
        self._bvh = _BVHNode.build(self, ids)
        return self._bvh

    def intersect(self, origin, direction, max_t=1e9):
        if self._bvh is None:
            self.build_bvh()
        hit = [None]
        self._bvh.intersect(self, origin, direction, max_t, hit)
        return hit[0]


class _BVHNode:
    __slots__ = ("bmin", "bmax", "left", "right", "faces")

    def __init__(self):
        self.bmin = None
        self.bmax = None
        self.left = None
        self.right = None
        self.faces = None

    @classmethod
    def build(cls, mesh, ids):
        node = cls()
        bmin = [1e30] * 3
        bmax = [-1e30] * 3
        # Bounding boxes must span the actual triangle vertices, not just
        # centroids: a large wall triangle has a centroid at one point but
        # vertices across the whole surface, and a centroid-only box would
        # be degenerate and reject valid rays.
        cmin = [1e30] * 3
        cmax = [-1e30] * 3
        for fid in ids:
            i0, i1, i2, _ = mesh.faces[fid]
            for vi in (i0, i1, i2):
                p = mesh.verts[vi]
                for k in (0, 1, 2):
                    if p[k] < bmin[k]:
                        bmin[k] = p[k]
                    if p[k] > bmax[k]:
                        bmax[k] = p[k]
            c = mesh._centroid[fid]
            for k in (0, 1, 2):
                if c[k] < cmin[k]:
                    cmin[k] = c[k]
                if c[k] > cmax[k]:
                    cmax[k] = c[k]
        node.bmin = tuple(bmin)
        node.bmax = tuple(bmax)
        if len(ids) <= 4:
            node.faces = tuple(ids)
            return node
        extent = [cmax[k] - cmin[k] for k in range(3)]
        axis = 0 if extent[0] >= extent[1] and extent[0] >= extent[2] else (
            1 if extent[1] >= extent[2] else 2)
        mid = 0.5 * (cmin[axis] + cmax[axis])
        left_ids = [f for f in ids if mesh._centroid[f][axis] < mid]
        right_ids = [f for f in ids if mesh._centroid[f][axis] >= mid]
        if not left_ids or not right_ids:  # fall back to median split
            ids.sort(key=lambda f: mesh._centroid[f][axis])
            cut = len(ids) // 2
            left_ids, right_ids = ids[:cut], ids[cut:]
        node.left = cls.build(mesh, left_ids)
        node.right = cls.build(mesh, right_ids)
        return node

    def _hit_bbox(self, origin, inv_dir, max_t):
        # tmin starts at -infinity, NOT 0: a ray origin inside the box (or
        # past its near slab) must still register the box as intersected.
        tmin = -1e30
        tmax = max_t
        for k in range(3):
            if abs(inv_dir[k]) > 1e30:
                if origin[k] < self.bmin[k] or origin[k] > self.bmax[k]:
                    return False
                continue
            t1 = (self.bmin[k] - origin[k]) * inv_dir[k]
            t2 = (self.bmax[k] - origin[k]) * inv_dir[k]
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
        return True

    def intersect(self, mesh, origin, direction, max_t, hit):
        inv_dir = (1.0 / direction[0] if direction[0] != 0 else 1e33,
                   1.0 / direction[1] if direction[1] != 0 else 1e33,
                   1.0 / direction[2] if direction[2] != 0 else 1e33)
        if not self._hit_bbox(origin, inv_dir, max_t):
            return
        if self.faces is not None:
            for fid in self.faces:
                t = _intersect_triangle(mesh, fid, origin, direction, max_t)
                if t is not None:
                    max_t = t[0]
                    hit[0] = t
            return
        self.left.intersect(mesh, origin, direction, max_t, hit)
        if hit[0] is not None:
            max_t = hit[0][0]
        self.right.intersect(mesh, origin, direction, max_t, hit)


def _intersect_triangle(mesh, fid, origin, direction, max_t):
    v0 = mesh._v0[fid]
    e1 = mesh._e1[fid]
    e2 = mesh._e2[fid]
    pvec = vec.cross(direction, e2)
    det = vec.dot(e1, pvec)
    if -1e-10 < det < 1e-10:
        return None
    inv_det = 1.0 / det
    tvec = vec.sub(origin, v0)
    u = vec.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qvec = vec.cross(tvec, e1)
    v = vec.dot(direction, qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = vec.dot(e2, qvec) * inv_det
    if t <= _EPS or t >= max_t:
        return None
    return (t, fid, u, v)
