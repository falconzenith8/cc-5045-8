"""Zero-dependency PNG output and display-only tone mapping.

The baker stores linear radiance in JSON block files. PNGs are produced from
that linear data with a Reinhard operator *at export time*; the energy units
in the stored data are never altered by display transforms.
"""
import math
import struct
import zlib

from . import vec


def reinhard(rgb, exposure=1.0):
    return tuple((c * exposure) / (1.0 + c * exposure) for c in rgb)


def _srgb(c):
    c = max(0.0, min(1.0, c))
    return round(12.92 * c * 255) if c <= 0.0031308 else round(
        (1.055 * c ** (1.0 / 2.4) - 0.055) * 255)


def linear_to_bytes(rgb, exposure=1.0):
    m = reinhard(rgb, exposure)
    return bytes(_srgb(c) for c in m)


def write_png(path, width, height, rgba_rows):
    """rgba_rows: list/iterable of bytes, width*4 RGBA per row."""
    raw = bytearray()
    for row in rgba_rows:
        raw.append(0)  # filter type 0
        raw.extend(row)

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


MISSING_MAGENTA = (1.0, 0.0, 1.0)
BACKGROUND = (0.02, 0.02, 0.025)


def _median_patch(values, atlas, idx, radius):
    """Per-channel median of same-chart texels in a (2r+1) window."""
    s = atlas.size
    x, y = idx % s, idx // s
    cid = atlas.chart_id[idx]
    chans = [[], [], []]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < s and 0 <= ny < s:
                nidx = ny * s + nx
                if nidx in values and atlas.chart_id[nidx] == cid:
                    nrgb = values[nidx]
                    for k in range(3):
                        chans[k].append(nrgb[k])
    if len(chans[0]) < 5:
        return None
    return tuple(sorted(chans[k])[len(chans[k]) // 2] for k in range(3))


def _suppress_outliers(atlas, values, threshold=2.2, radius=2, passes=2):
    """Iterative same-chart median filter: dense Monte-Carlo noise clusters
    are pulled toward the local median while genuine edges (which persist in
    the median) survive. No texel outside the chart is ever read."""
    cur = dict(values)
    for _ in range(passes):
        nxt = dict(cur)
        for idx, rgb in cur.items():
            if not atlas.valid[idx] or atlas.dilated[idx] == 2:
                continue
            med = _median_patch(cur, atlas, idx, radius)
            if med is None:
                continue
            cur_lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            med_lum = 0.2126 * med[0] + 0.7152 * med[1] + 0.0722 * med[2]
            if med_lum > 1e-6 and cur_lum > threshold * med_lum:
                nxt[idx] = med
        cur = nxt
    return cur


def edge_aware_denoise(atlas, values, iterations=3, spatial=2.0,
                       color_sigma=0.35):
    """Edge-preserving bilateral filter restricted to each chart.

    Operates only on assembled values for display/export: the block JSON
    files keep the unbiased per-texel linear radiance. A neighbour only
    contributes when it belongs to the same chart, so chart padding and
    separate surfaces can never bleed into one another. The colour gate is
    scaled by local luminance, so dark regions get smoothed as strongly as
    bright ones while genuine reflectivity/shadow edges stay sharp. An
    initial local-median pass removes isolated Monte-Carlo outliers.
    """
    s = atlas.size
    cur = dict(values)
    cur = _suppress_outliers(atlas, cur)
    radius = 2
    spatial_w = [math.exp(-(dx * dx + dy * dy) / (2.0 * spatial * spatial))
                 for dy in range(-radius, radius + 1)
                 for dx in range(-radius, radius + 1)]
    for _ in range(iterations):
        nxt = dict(cur)
        for idx, rgb in cur.items():
            if not atlas.valid[idx] or atlas.dilated[idx] == 2:
                continue
            x, y = idx % s, idx // s
            cid = atlas.chart_id[idx]
            local_lum = [0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]]
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < s and 0 <= ny < s:
                        nidx = ny * s + nx
                        if nidx in cur and atlas.chart_id[nidx] == cid:
                            n = cur[nidx]
                            local_lum.append(
                                0.2126 * n[0] + 0.7152 * n[1]
                                + 0.0722 * n[2])
            local_lum.sort()
            sigma_eff = color_sigma * max(local_lum[len(local_lum) // 2],
                                          0.02)
            acc = [0.0, 0.0, 0.0]
            wsum = 0.0
            wi = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < s and 0 <= ny < s):
                        continue
                    nidx = ny * s + nx
                    if nidx not in cur or atlas.chart_id[nidx] != cid:
                        wi += 1
                        continue
                    nrgb = cur[nidx]
                    cdiff = sum((rgb[k] - nrgb[k]) ** 2 for k in range(3))
                    cw = math.exp(-cdiff / (2.0 * sigma_eff * sigma_eff))
                    w = spatial_w[wi] * cw
                    for k in range(3):
                        acc[k] += nrgb[k] * w
                    wsum += w
                    wi += 1
            if wsum > 0.0:
                nxt[idx] = (acc[0] / wsum, acc[1] / wsum, acc[2] / wsum)
        cur = nxt
    return cur


def render_atlas_png(path, atlas, values, missing=None, exposure=0.35,
                     block_size=32, draw_blocks=False):
    """Assemble final atlas PNG. `values` maps texel index -> linear rgb.

    Missing covered texels (failed/never-run blocks) are painted magenta so
    the preview clearly shows unfinished regions. Dilation texels inherit the
    nearest baked texel of their own chart during assembly.
    """
    s = atlas.size
    missing = set(missing or ())
    filled = {}
    for idx, rgb in values.items():
        filled[idx] = rgb
    # dilation fill: repeated nearest-neighbour passes inside each chart
    for _ in range(6):
        for idx in range(s * s):
            if idx in filled or not atlas.valid[idx]:
                continue
            x, y = idx % s, idx // s
            cid = atlas.chart_id[idx]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < s and 0 <= ny < s):
                    continue
                nidx = ny * s + nx
                if nidx in filled and atlas.chart_id[nidx] == cid:
                    filled[idx] = filled[nidx]
                    break
    rows = []
    for y in range(s):
        row = bytearray(s * 4)
        for x in range(s):
            idx = y * s + x
            off = x * 4
            if not atlas.valid[idx]:
                rgb = BACKGROUND
                a = 255
            elif idx in missing:
                rgb = MISSING_MAGENTA
                a = 255
            elif idx in filled:
                rgb = filled[idx]
                a = 255
            else:
                rgb = MISSING_MAGENTA  # covered but never computed
                a = 255
            b = linear_to_bytes(rgb, exposure)
            row[off:off + 3] = b
            row[off + 3] = a
        if draw_blocks:
            for x in range(s):
                if x % block_size == 0:
                    row[x * 4:x * 4 + 4] = bytes((0, 255, 200, 255))
        rows.append(bytes(row))
    if draw_blocks:
        for by in range(0, s, block_size):
            row = bytearray(rows[by])
            for x in range(s):
                row[x * 4:x * 4 + 4] = bytes((0, 255, 200, 255))
            rows[by] = bytes(row)
    write_png(path, s, s, rows)


def render_mask_png(path, atlas):
    """Coverage mask: white = real coverage, yellow = dilation, black = gap."""
    s = atlas.size
    rows = []
    for y in range(s):
        row = bytearray(s * 4)
        for x in range(s):
            idx = y * s + x
            off = x * 4
            if not atlas.valid[idx]:
                c = (8, 8, 10)
            elif atlas.dilated[idx] == 2:
                c = (220, 190, 40)
            else:
                c = (235, 235, 235)
            row[off:off + 3] = bytes(c)
            row[off + 3] = 255
        rows.append(bytes(row))
    write_png(path, s, s, rows)


def render_chart_id_png(path, atlas):
    """Debug chart-id map: each chart gets a stable distinct colour."""
    palette = [(200, 70, 70), (70, 150, 200), (90, 180, 110),
               (210, 160, 60), (160, 100, 200), (90, 200, 200),
               (220, 120, 170), (150, 150, 80)]
    s = atlas.size
    rows = []
    for y in range(s):
        row = bytearray(s * 4)
        for x in range(s):
            idx = y * s + x
            off = x * 4
            cid = atlas.chart_id[idx]
            c = (10, 10, 12) if cid < 0 else palette[cid % len(palette)]
            row[off:off + 3] = bytes(c)
            row[off + 3] = 255
        rows.append(bytes(row))
    write_png(path, s, s, rows)
