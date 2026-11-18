"""Minimal 3D vector math on plain tuples (no third-party dependency)."""
import math


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vmul(a, b):
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    l = length(a)
    if l <= 1e-12:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / l
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def reflect(d, n):
    k = 2.0 * dot(n, d)
    return (d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2])


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def clampv(v, hi=1.0):
    return (clamp(v[0], 0.0, hi), clamp(v[1], 0.0, hi), clamp(v[2], 0.0, hi))
