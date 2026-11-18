"""Baked test scenes.

All geometry is emitted in metres; emissive materials carry linear radiance
in W/(m^2 sr). Nothing here comes from a screen capture - every scene is
analytic geometry rendered by the path-tracing kernel.

Scenes:
  closed_room: sealed box room, ceiling area light, tinted wall (indirect
               colour bleed), interior pillar.
  thin_wall:   two rooms separated by a single-layer (zero thickness) wall.
               The light is in one room only; the other room must stay dark.
  seam_floor:  floor made of four disconnected slabs with open gaps over a
               dark base. Adjacent slabs get separate atlas charts.
"""

W, H, D = 8.0, 3.0, 8.0


def _closed_shell(mesh, wall_mat, floor_mat, ceil_mat, half=4.0, height=H):
    """Walls, floor and ceiling of a sealed room centred on the origin."""
    h = half
    # floor and ceiling
    mesh.add_quad((-h, 0, -h), (-h, 0, h), (h, 0, h), (h, 0, -h), floor_mat)
    mesh.add_quad((-h, height, h), (-h, height, -h),
                  (h, height, -h), (h, height, h), ceil_mat)
    # four walls
    mesh.add_quad((-h, 0, -h), (h, 0, -h), (h, height, -h),
                  (-h, height, -h), wall_mat)            # north
    mesh.add_quad((h, 0, h), (-h, 0, h), (-h, height, h),
                  (h, height, h), wall_mat)              # south
    mesh.add_quad((-h, 0, h), (-h, 0, -h), (-h, height, -h),
                  (-h, height, h), wall_mat)             # west
    mesh.add_quad((h, 0, -h), (h, 0, h), (h, height, h),
                  (h, height, -h), wall_mat)             # east


def _ceiling_light(mesh, light_mat, cx, cz, size=1.6, y=H - 0.06):
    half = size * 0.5
    mesh.add_quad((cx - half, y, cz - half), (cx - half, y, cz + half),
                  (cx + half, y, cz + half), (cx + half, y, cz - half),
                  light_mat)


def build_closed_room():
    mesh = __import__("lightbaker.geometry", fromlist=["Mesh"]).Mesh("closed_room")
    wall = mesh.add_material("wall_white", albedo=(0.78, 0.78, 0.76))
    red = mesh.add_material("wall_red", albedo=(0.62, 0.16, 0.14))
    floor = mesh.add_material("floor_grey", albedo=(0.45, 0.44, 0.42))
    ceil = mesh.add_material("ceiling", albedo=(0.82, 0.82, 0.80))
    pillar = mesh.add_material("pillar", albedo=(0.30, 0.42, 0.55))
    light = mesh.add_material("ceiling_light", albedo=(1, 1, 1),
                              emission=(17.0, 15.5, 12.5))
    _closed_shell(mesh, wall, floor, ceil)
    # repaint the west wall red so indirect colour bleed is visible
    west_faces = (6, 7)
    for fid in west_faces:
        i0, i1, i2, _ = mesh.faces[fid]
        mesh.faces[fid] = (i0, i1, i2, red)
    # interior pillar creating contact shadows and multi-bounce occlusion
    mesh.add_box((1.4, 0.0, 1.2), (2.3, 1.7, 2.1), pillar)
    _ceiling_light(mesh, light, cx=-1.0, cz=-0.5)
    mesh.build_bvh()
    return mesh, {
        "name": "closed_room",
        "description": "Sealed 8x3x8 m room, one ceiling area light, "
                       "red west wall for indirect colour bleed, pillar.",
        "units": "radiance W/(m^2 sr), linear sRGB primaries",
    }


def build_thin_wall():
    from .geometry import Mesh
    mesh = Mesh("thin_wall")
    wall = mesh.add_material("wall_white", albedo=(0.78, 0.78, 0.76))
    floor = mesh.add_material("floor_grey", albedo=(0.45, 0.44, 0.42))
    ceil = mesh.add_material("ceiling", albedo=(0.82, 0.82, 0.80))
    partition = mesh.add_material("thin_partition", albedo=(0.72, 0.70, 0.55))
    light = mesh.add_material("left_room_light", albedo=(1, 1, 1),
                              emission=(18.0, 17.0, 14.0))
    _closed_shell(mesh, wall, floor, ceil)
    # Thin (6 cm) solid partition at x=0. The two faces have opposite
    # normals and therefore become separate atlas charts with separate
    # texels: the dark-room side must bake dark even though the lit-room
    # side is bright. The small thickness still blocks every ray, so no
    # geometric light leak is possible.
    t = 0.03
    z = 4.0 - 1e-4
    mesh.add_box((-t, 0.0, -z), (t, H, z), partition)
    _ceiling_light(mesh, light, cx=-2.2, cz=0.0, size=1.4)
    mesh.build_bvh()
    return mesh, {
        "name": "thin_wall",
        "description": "Sealed room split by a 6 cm thin partition whose "
                       "opposite faces are separate charts. Lamp in the "
                       "left room only; right room must stay dark "
                       "(light-leak check).",
        "units": "radiance W/(m^2 sr), linear sRGB primaries",
    }


def build_seam_floor():
    from .geometry import Mesh
    mesh = Mesh("seam_floor")
    wall = mesh.add_material("wall_white", albedo=(0.78, 0.78, 0.76))
    ceil = mesh.add_material("ceiling", albedo=(0.82, 0.82, 0.80))
    slab_a = mesh.add_material("slab_a", albedo=(0.50, 0.48, 0.45))
    slab_b = mesh.add_material("slab_b", albedo=(0.38, 0.40, 0.44))
    base = mesh.add_material("dark_base", albedo=(0.04, 0.04, 0.045))
    light = mesh.add_material("ceiling_light", albedo=(1, 1, 1),
                              emission=(16.0, 15.0, 12.5))
    # shell without the floor (floor is rebuilt as separate slabs)
    h = 4.0
    mesh.add_quad((-h, H, h), (-h, H, -h), (h, H, -h), (h, H, h), ceil)
    mesh.add_quad((-h, 0, -h), (h, 0, -h), (h, H, -h), (-h, H, -h), wall)
    mesh.add_quad((h, 0, h), (-h, 0, h), (-h, H, h), (h, H, h), wall)
    mesh.add_quad((-h, 0, h), (-h, 0, -h), (-h, H, -h), (-h, H, h), wall)
    mesh.add_quad((h, 0, -h), (h, 0, h), (h, H, h), (h, H, -h), wall)
    # dark recessed base under the whole floor
    mesh.add_quad((-h, -0.08, -h), (-h, -0.08, h),
                  (h, -0.08, h), (h, -0.08, -h), base)
    # four disconnected floor slabs separated by 0.12 m open gaps
    gap = 0.12
    inner = h - 0.3
    step = inner
    bounds = [(-inner, -gap / 2), (gap / 2, inner)]
    slab_index = 0
    for x0, x1 in bounds:
        for z0, z1 in bounds:
            mat = slab_a if slab_index % 2 == 0 else slab_b
            mesh.add_quad((x0, 0, z0), (x0, 0, z1),
                          (x1, 0, z1), (x1, 0, z0), mat)
            slab_index += 1
    _ceiling_light(mesh, light, cx=0.0, cz=0.0, size=1.8)
    mesh.build_bvh()
    return mesh, {
        "name": "seam_floor",
        "description": "Room floor = four disjoint slabs with open gaps over "
                       "a dark base. Adjacent slabs are separate atlas "
                       "charts and must not bleed into each other.",
        "units": "radiance W/(m^2 sr), linear sRGB primaries",
    }


SCENES = {
    "closed_room": build_closed_room,
    "thin_wall": build_thin_wall,
    "seam_floor": build_seam_floor,
}


def build(name):
    if name not in SCENES:
        raise KeyError("unknown scene %r; choose from %s"
                       % (name, sorted(SCENES)))
    return SCENES[name]()
