"""The gas and foam in a shaken bottle of sparkling water: a small model, pure Python, shared by the task and the demo.

State: headspace pressure P (bar, gauge), dissolved-gas pressure D (bar: the pressure the water would come to
equilibrium at), foam volume F (mL) on the water, and the bottle's nucleation factor N (shaking leaves bubble nuclei
that make gas leave the water fast; they fade over minutes).

    e    = K_E * N * max(D - P, 0)            gas leaving the water (bar/s of headspace pressure)
    dP/dt = e - P * open(theta) / TAU_LEAK     the cap's seal leaks once it is turned past THETA_SEAL
    dD/dt = -R_D * e                           the water loses gas as it comes out
    dF/dt = C_F * N * e - F / TAU_F            bubbles make foam; foam collapses
    N    = 1 + (N_SHAKEN - 1) * exp(-t / TAU_N)

So a cap twisted straight off drops P at once, the water gushes gas (D - P large, N high) and the foam climbs out of
the neck; a cap turned only until it hisses, and turned back when the foam climbs, lets the gas out a little at a
time while the foam collapses between.
"""

from __future__ import annotations

import math

P0 = 2.2              # bar: headspace and dissolved gas at the start (chilled, carbonated, sealed)
K_E = 0.35            # 1/s per unit N
R_D = 0.30            # dissolved gas lost per unit of gas out
TAU_LEAK = 0.20       # s: the headspace's time constant with the gas path fully open
THETA_SEAL = math.radians(45.0)   # the seal opens here
THETA_FULL = math.radians(240.0)  # the gas path is fully open here
THETA_OFF = math.radians(560.0)   # the thread lets go here (about 1.55 turns)
C_F = 9.0             # mL of foam per bar of gas out, per unit N
TAU_F = 2.5           # s: foam collapses
N_SHAKEN, TAU_N = 7.0, 240.0
F_SHAKEN = 8.0        # mL of foam on the water just after shaking
VENTED = 0.05         # bar: below this the hiss has stopped

# the bottle's inside, from its base (m): the water's surface, and the volume above it to each height (mL)
Z_WATER = 0.150
PROFILE = [(0.150, 0.0315), (0.160, 0.0300), (0.172, 0.0240), (0.182, 0.0150), (0.190, 0.0105), (0.205, 0.0105)]
Z_RIM = 0.205


def opening(theta: float) -> float:
    if theta <= THETA_SEAL:
        return 0.0
    return min(1.0, (theta - THETA_SEAL) / (THETA_FULL - THETA_SEAL))


def volume_to(z: float) -> float:
    """mL of the bottle's inside between the water's surface and height z."""
    v, (z0, r0) = 0.0, PROFILE[0]
    for z1, r1 in PROFILE[1:]:
        if z <= z0:
            break
        top = min(z, z1)
        rr = r0 + (r1 - r0) * (top - z0) / (z1 - z0)
        v += math.pi * (r0 * r0 + r0 * rr + rr * rr) / 3 * (top - z0) * 1e6
        z0, r0 = z1, r1
    return v


V_RIM = volume_to(Z_RIM)


def foam_top(F: float) -> float:
    """Height of the foam's top (m from the base) for F mL of foam."""
    if F <= 0:
        return Z_WATER
    lo, hi = Z_WATER, Z_RIM + 0.05
    if F >= V_RIM:
        return Z_RIM + (F - V_RIM) * 1e-6 / (math.pi * 0.0105 ** 2)
    for _ in range(40):
        mid = (lo + hi) / 2
        if volume_to(mid) < F:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class Bottle:
    def __init__(self, shaken: bool = True):
        self.P = self.D = P0
        self.F = F_SHAKEN if shaken else 0.0
        self.t = 0.0
        self.n0 = N_SHAKEN if shaken else 1.0
        self.max_top = Z_WATER

    @property
    def N(self) -> float:
        return 1.0 + (self.n0 - 1.0) * math.exp(-self.t / TAU_N)

    def step(self, theta: float, dt: float, off: bool = False) -> None:
        sub = max(1, int(dt / 0.002))
        h = dt / sub
        for _ in range(sub):
            a = 1.0 if off else opening(theta)
            e = K_E * self.N * max(self.D - self.P, 0.0)
            self.P = max(0.0, self.P + h * (e - self.P * a / TAU_LEAK))
            self.D = max(0.0, self.D - h * R_D * e)
            self.F = max(0.0, self.F + h * (C_F * self.N * e - self.F / TAU_F))
            self.t += h
        self.max_top = max(self.max_top, foam_top(self.F))

    @property
    def top(self) -> float:
        return foam_top(self.F)

    @property
    def vented(self) -> bool:
        return self.P < VENTED and K_E * self.N * max(self.D - self.P, 0.0) < 0.02
