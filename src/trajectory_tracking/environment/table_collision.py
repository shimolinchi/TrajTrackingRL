"""Deterministic flat-capped circular table collision geometry (metres)."""

import math

TABLE_COLLISION_SEGMENTS = 128


def circular_table_mesh(radius: float, height: float):
    """Closed outward-facing triangle mesh; visual assets are not modified."""
    if not all(math.isfinite(x) and x > 0 for x in (radius, height)):
        raise ValueError("table radius and height must be finite and positive")
    n = TABLE_COLLISION_SEGMENTS
    ring = [(radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
    points = [(x, y, z) for z in (-height / 2, height / 2) for x, y in ring]
    points += [(0.0, 0.0, -height / 2), (0.0, 0.0, height / 2)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((2*n, j, i), (2*n+1, n+i, n+j),
                      (i, j, n+j), (i, n+j, n+i)))
    return points, faces
