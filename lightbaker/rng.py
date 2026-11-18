"""Deterministic, auditable random number generation.

PCG32 (Melissa O'Neill, PCG, 2014). Every stochastic decision in the baker is
driven by a PCG32 stream whose seed is derived from (job_seed, block index,
texel index, sample index) via BLAKE2b. Re-running a block with the same
parameters reproduces bit-identical output regardless of which worker process
or retry attempt computes it.
"""
import hashlib

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
_MULT = 6364136223846793005
_INC = 1442695040888963407


def derive_seed(*parts) -> int:
    """Map arbitrary integer/string parts to a reproducible 64-bit seed."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little")


class PCG32:
    __slots__ = ("state", "inc")

    def __init__(self, seed: int, seq: int = 0):
        self.inc = ((seq << 1) | 1) & MASK64
        self.state = 0
        self._advance()
        self.state = (self.state + (seed & MASK64)) & MASK64
        self._advance()

    def _advance(self):
        self.state = (self.state * _MULT + self.inc) & MASK64

    def next_u32(self) -> int:
        old = self.state
        self._advance()
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32

    def uniform(self) -> float:
        """Uniform float in [0, 1) with 32 bits of mantissa."""
        return self.next_u32() / float(1 << 32)

    def uniform2(self) -> float:
        """Uniform float in [0, 1) with 53 bits of mantissa."""
        hi = self.next_u32()
        lo = self.next_u32()
        return ((hi << 21) ^ (lo >> 11)) / float(1 << 53)

    def uniform_range(self, a: float, b: float) -> float:
        return a + (b - a) * self.uniform()

    def cosine_hemisphere(self):
        """Cosine-weighted sample on the unit hemisphere. Returns (x,y,z,pdf)."""
        u1 = self.uniform2()
        u2 = self.uniform2()
        r = u1 ** 0.5
        phi = 6.283185307179586 * u2
        x = r * _cos(phi)
        y = r * _sin(phi)
        z = (max(0.0, 1.0 - u1)) ** 0.5
        return (x, y, z, z * 0.3183098861837907)

    def uniform_triangle(self):
        """Uniform barycentric (u, v) with u+v <= 1 for a unit triangle."""
        u1 = self.uniform2()
        u2 = self.uniform2()
        if u1 + u2 > 1.0:
            u1 = 1.0 - u1
            u2 = 1.0 - u2
        return u1, u2


import math as _m

_cos = _m.cos
_sin = _m.sin
