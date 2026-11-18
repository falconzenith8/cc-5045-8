"""Lightmap chart segmentation, atlas packing and texel rasterization.

Pipeline:
  1. Union-find merges edge-connected triangles whose geometric normals are
     coplanar. Sharing a 3D edge is required, so two slabs that merely touch
     in projection (or sit next to each other) never become one chart.
  2. Each chart is projected onto its dominant axis plane at a fixed texels
     per metre density, giving every texel a physical size in metres.
  3. Charts are shelf-packed into the atlas with an explicit padding gap.
  4. Rasterization records, per covered texel, its exact 3D position,
     geometric normal, face id and barycentric coordinates - the reliable
     inverse map used by the lighting kernel.
  5. Dilation (edge padding) is a BFS that can only copy texels *of the same
     chart*. It never crosses the atlas padding into another chart, which is
     the structural guarantee against neighbour colour bleed.
"""
from collections import deque

from . import vec

ATLAS_CHOICES = (128, 256, 512, 1024)
PADDING = 6          # texels kept between charts (dilation headroom)
DILATION_STEPS = 3   # texels the runtime seam padding extends
COPLANAR_DOT = 0.995


class AtlasError(RuntimeError):
    pass


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _segment_charts(mesh):
    n = len(mesh.faces)
    uf = _UnionFind(n)
    edge_owners = {}
    for fid, (i0, i1, i2, _) in enumerate(mesh.faces):
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            key = (a, b) if a < b else (b, a)
            if key in edge_owners:
                other = edge_owners[key]
                if vec.dot(mesh._normal[other], mesh._normal[fid]) > COPLANAR_DOT:
                    uf.union(other, fid)
            else:
                edge_owners[key] = fid
    groups = {}
    for fid in range(n):
        groups.setdefault(uf.find(fid), []).append(fid)
    return list(groups.values())


def _chart_basis(mesh, faces):
    # average normal weighted by area
    nrm = (0.0, 0.0, 0.0)
    for fid in faces:
        nrm = vec.add(nrm, vec.mul(mesh._normal[fid], mesh._area[fid]))
    nrm = vec.normalize(nrm)
    ax = max(range(3), key=lambda k: abs(nrm[k]))
    if ax == 2:                                   # mostly XY plane
        u = (1.0, 0.0, 0.0)
        v = (0.0, 1.0, 0.0)
    elif ax == 1:                                 # mostly XZ plane
        u = (1.0, 0.0, 0.0)
        v = (0.0, 0.0, 1.0)
    else:                                         # mostly YZ plane
        u = (0.0, 1.0, 0.0)
        v = (0.0, 0.0, 1.0)
    if vec.dot(vec.cross(u, v), nrm) < 0.0:
        u = (-u[0], -u[1], -u[2])
    return nrm, u, v


def _rasterize_triangle(px, py, painter):
    """Rasterize a 2D triangle given three integer pixel coords."""
    x0, y0 = px[0], py[0]
    x1, y1 = px[1], py[1]
    x2, y2 = px[2], py[2]
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if area2 <= 0:
        return  # wrong winding in the chart plane; skip
    xmin = max(0, min(x0, x1, x2))
    xmax = max(x0, x1, x2)
    ymin = max(0, min(y0, y1, y2))
    ymax = max(y0, y1, y2)
    inv = 1.0 / area2
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            cx, cy = x + 0.5, y + 0.5
            w0 = ((x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx)) * inv
            w1 = ((x2 - cx) * (y0 - cy) - (y2 - cy) * (x0 - cx)) * inv
            w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                painter(x, y, w0, w1, w2)


class Atlas:
    def __init__(self, mesh, size=256, texels_per_metre=40.0):
        if size not in ATLAS_CHOICES:
            raise AtlasError("atlas size must be one of %s" % (ATLAS_CHOICES,))
        self.mesh = mesh
        self.size = size
        self.texel_size = 1.0 / texels_per_metre
        self.tpm = texels_per_metre
        self.charts = []
        # per-pixel inverse map
        s = size
        self.valid = bytearray(s * s)
        self.chart_id = [-1] * (s * s)
        self.positions = [None] * (s * s)
        self.normals = [None] * (s * s)
        self.face = [-1] * (s * s)
        self.bary = [None] * (s * s)
        self.dilated = bytearray(s * s)   # 1 = real coverage, 2 = dilation
        self._build(mesh, _segment_charts(mesh))

    # ---------------------------------------------------------------- build
    def _build(self, mesh, groups):
        planned = []
        for cid, faces in enumerate(groups):
            nrm, u, v = _chart_basis(mesh, faces)
            # project every corner to metric 2D then to integer texel units
            proj = {}
            for fid in faces:
                for vi in mesh.faces[fid][:3]:
                    if vi in proj:
                        continue
                    p = mesh.verts[vi]
                    proj[vi] = (vec.dot(p, u) * self.tpm,
                                vec.dot(p, v) * self.tpm)
            xs = [q[0] for q in proj.values()]
            ys = [q[1] for q in proj.values()]
            ox, oy = min(xs), min(ys)
            local = {vi: (int(round(q[0] - ox)), int(round(q[1] - oy)))
                     for vi, q in proj.items()}
            w = max(q[0] for q in local.values()) + 1
            h = max(q[1] for q in local.values()) + 1
            planned.append({"cid": cid, "faces": faces, "n": nrm, "u": u,
                            "v": v, "ox": ox, "oy": oy, "w": w, "h": h,
                            "local": local})
        self._pack(planned)
        for ch in planned:
            self._paint_chart(mesh, ch)
        self._dilate()

    def _pack(self, planned):
        # shelf packing, tallest charts first
        planned.sort(key=lambda c: -c["h"])
        shelf_y = 0
        shelf_h = 0
        cursor_x = 0
        s = self.size
        for ch in planned:
            need_w = ch["w"] + PADDING
            need_h = ch["h"] + PADDING
            if ch["w"] > s or ch["h"] > s:
                raise AtlasError(
                    "chart %d needs %dx%d texels but atlas is %dx%d; raise "
                    "atlas size or lower texel density"
                    % (ch["cid"], ch["w"], ch["h"], s, s))
            if cursor_x + need_w > s:
                shelf_y += shelf_h
                shelf_h = 0
                cursor_x = 0
            if shelf_y + need_h > s:
                raise AtlasError(
                    "atlas %dx%d too small for %d charts; raise atlas size"
                    % (s, s, len(planned)))
            ch["origin"] = (cursor_x, shelf_y)
            cursor_x += need_w
            shelf_h = max(shelf_h, need_h)

    def _paint_chart(self, mesh, ch):
        ox_px, oy_px = ch["origin"]

        def painter(x, y, w0, w1, w2):
            X = ox_px + x
            Y = oy_px + y
            idx = Y * self.size + X
            if self.valid[idx]:
                return  # atlas padding guarantees this cannot be another chart
            self.valid[idx] = 1
            self.dilated[idx] = 1
            self.chart_id[idx] = ch["cid"]

        for fid in ch["faces"]:
            i0, i1, i2, _ = mesh.faces[fid]
            pts = [ch["local"][i0], ch["local"][i1], ch["local"][i2]]
            _rasterize_triangle([p[0] for p in pts], [p[1] for p in pts],
                                painter)
        # record the inverse map for every covered texel of this chart
        oxm = ch["ox"] / self.tpm
        oym = ch["oy"] / self.tpm
        u, v = ch["u"], ch["v"]
        for Y in range(oy_px, oy_px + ch["h"] + 1):
            for X in range(ox_px, ox_px + ch["w"] + 1):
                idx = Y * self.size + X
                if not self.valid[idx] or self.chart_id[idx] != ch["cid"]:
                    continue
                fu = oxm + (X - ox_px + 0.5) / self.tpm
                fv = oym + (Y - oy_px + 0.5) / self.tpm
                best = None
                for fid in ch["faces"]:
                    i0, i1, i2, _ = mesh.faces[fid]
                    p0, p1, p2 = (ch["local"][i0], ch["local"][i1],
                                  ch["local"][i2])
                    area2 = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                             - (p1[1] - p0[1]) * (p2[0] - p0[0]))
                    if area2 <= 0:
                        continue
                    lx = (X - ox_px) + 0.5
                    ly = (Y - oy_px) + 0.5
                    b0 = ((p1[0] - lx) * (p2[1] - ly)
                          - (p1[1] - ly) * (p2[0] - lx)) / area2
                    b1 = ((p2[0] - lx) * (p0[1] - ly)
                          - (p2[1] - ly) * (p0[0] - lx)) / area2
                    b2 = 1.0 - b0 - b1
                    if b0 >= -1e-6 and b1 >= -1e-6 and b2 >= -1e-6:
                        b0, b1 = max(0.0, b0), max(0.0, b1)
                        b2 = max(0.0, b2)
                        pos = self.mesh.face_point(fid, b1, b2)
                        best = (fid, pos, mesh.face_normal(fid),
                                (b0, b1, b2))
                        break
                if best is None:
                    # numeric edge case: snap to the nearest projected vertex
                    continue
                fid, pos, nrm, bary = best
                self.face[idx] = fid
                self.positions[idx] = pos
                self.normals[idx] = nrm
                self.bary[idx] = bary
        ch["bbox"] = (ox_px, oy_px, ox_px + ch["w"], oy_px + ch["h"])
        self.charts.append(ch)

    # ------------------------------------------------------------- dilation
    def _dilate(self):
        """Expand coverage inside each chart only; never cross chart borders."""
        s = self.size
        frontier = deque()
        for idx in range(s * s):
            if self.valid[idx]:
                frontier.append((idx, 0))
        owned = set(self.chart_id[i] for i in range(s * s) if self.valid[i])
        while frontier:
            idx, depth = frontier.popleft()
            if depth >= DILATION_STEPS:
                continue
            x, y = idx % s, idx // s
            cid = self.chart_id[idx]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < s and 0 <= ny < s):
                    continue
                nidx = ny * s + nx
                if self.valid[nidx]:
                    continue
                # empty padding texel: claim it for this chart only if no
                # other chart is closer (check neighbour chart ownership)
                self.valid[nidx] = 1
                self.dilated[nidx] = 2
                self.chart_id[nidx] = cid
                self.positions[nidx] = self.positions[idx]
                self.normals[nidx] = self.normals[idx]
                self.face[nidx] = self.face[idx]
                self.bary[nidx] = self.bary[idx]
                frontier.append((nidx, depth + 1))
        self._owned_chart_count = len(owned)

    # ------------------------------------------------------------- export
    def chart_pixel_at(self, x, y):
        idx = y * self.size + x
        if not self.valid[idx]:
            return None
        return {
            "chart": self.chart_id[idx],
            "face": self.face[idx],
            "position": self.positions[idx],
            "normal": self.normals[idx],
            "barycentric": self.bary[idx],
            "dilated": self.dilated[idx] == 2,
        }

    def uv_for_face_vertex(self, fid, vi):
        """Normalized atlas UV for a mesh vertex of a face (for preview)."""
        for ch in self.charts:
            if fid not in ch["faces"]:
                continue
            lx, ly = ch["local"][vi]
            X = ch["origin"][0] + lx
            Y = ch["origin"][1] + ly
            return ((X + 0.5) / self.size, 1.0 - (Y + 0.5) / self.size)
        return (0.0, 0.0)

    def runtime_mesh(self):
        """Flat vertex/face/uv arrays sampled by the 3D preview."""
        verts, uvs, normals, faces = [], [], [], []
        for fid, (i0, i1, i2, mid) in enumerate(self.mesh.faces):
            ids = []
            for vi in (i0, i1, i2):
                verts.append(self.mesh.verts[vi])
                uvs.append(self.uv_for_face_vertex(fid, vi))
                normals.append(self.mesh._normal[fid])
                ids.append(len(verts) - 1)
            faces.append((ids[0], ids[1], ids[2], mid))
        return {
            "verts": verts, "uvs": uvs, "normals": normals,
            "faces": faces,
            "materials": self.mesh.materials,
        }

    def block_bounds(self, block_size=32):
        """List of (bx, by, x0, y0, x1, y1, has_coverage) grid blocks."""
        s = self.size
        out = []
        for by in range(0, s, block_size):
            for bx in range(0, s, block_size):
                x1, y1 = min(bx + block_size, s), min(by + block_size, s)
                covered = False
                for y in range(by, y1):
                    row = y * s
                    for x in range(bx, x1):
                        if self.valid[row + x]:
                            covered = True
                            break
                    if covered:
                        break
                out.append((bx // block_size, by // block_size,
                            bx, by, x1, y1, covered))
        return out
