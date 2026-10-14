"""
Physical and astronomical constants (SI units).

Data layout
-----------
Every constant is a plain Python ``float`` (64-bit double precision). NumPy
broadcasts these scalars automatically when they multiply arrays.

Why double precision matters: in the Sun-Earth system positions are ~1e11 m
while the change per time step is ~1e2-1e4 m. A ``float32`` keeps only ~7
significant digits, so 1e11 + 1e3 would round back to 1e11 and the planet
would never move. ``float64`` keeps ~16 digits and the increment survives.
"""

# --------------------------------------------------------------------------
# Gravitational constant
# --------------------------------------------------------------------------
# CODATA 2018 value. The original MATLAB code used 6.67384e-11 (CODATA 2010);
# the 0.007 % difference is irrelevant for the dynamics.
G_SI: float = 6.67430e-11  # m^3 kg^-1 s^-2

# --------------------------------------------------------------------------
# Time and length units
# --------------------------------------------------------------------------
DAY: float = 86_400.0                  # s
YEAR: float = 365.25 * DAY             # Julian year, s (~3.156e7 s)
AU: float = 1.495_978_707e11           # astronomical unit (IAU 2012 definition), m

# --------------------------------------------------------------------------
# Masses (kg)
# --------------------------------------------------------------------------
M_SUN: float = 1.988_47e30
M_EARTH: float = 5.9722e24
M_MOON: float = 7.342e22

# --------------------------------------------------------------------------
# Earth-Moon orbital data (Sun-Earth-Moon scenario)
# --------------------------------------------------------------------------
EARTH_APHELION: float = 1.520_982_32e11   # m, Sun-Earth distance at aphelion
EARTH_APHELION_SPEED: float = 29_290.0    # m/s, Earth's orbital speed at aphelion
MOON_APOGEE: float = 4.055e8              # m, Earth-Moon distance at apogee
MOON_APOGEE_SPEED: float = 964.0          # m/s, Moon's speed relative to Earth at apogee
