# pylint: disable=no-name-in-module
"""Alpha shape boundary for 2-D point sets.

An alpha shape is a generalisation of the convex hull that allows concave
boundaries.  The ``alpha`` parameter controls how tightly the boundary
wraps around the data: larger values produce shapes closer to the convex
hull, while smaller values follow the data more tightly.

Typical usage::

    shape = AlphaShape(alpha=0.1).fit(train_points)
    outside, bdist = shape.query(test_points)
    polygon = shape.boundary_polygon()
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial import Delaunay


class AlphaShape:
    """Compute and query an alpha shape built from 2-D training points.

    Attributes:
        alpha: Alpha parameter (smaller = tighter hull).
        boundary_edges: List of ``(point_a, point_b)`` coordinate pairs
            that form the boundary after :meth:`fit`.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._train_points: np.ndarray = np.empty(0)
        self._inside_simplices: np.ndarray = np.empty(0)
        self._boundary_idx: list = []
        self.boundary_edges: list = []

    def fit(self, train_points: np.ndarray) -> "AlphaShape":
        """Build the alpha shape from training points.

        Args:
            train_points: (N, 2) array of training coordinates.

        Returns:
            Self, for chaining.
        """
        self._train_points = np.asarray(train_points, dtype=float)

        # Need at least 3 points to form a triangle in 2D
        if len(self._train_points) < self._train_points.shape[1] + 1:
            self._inside_simplices = np.empty((0, 3), dtype=int)
            self._boundary_idx = []
            self.boundary_edges = []
            return self

        # 1. Triangulate all training points (Delaunay = convex triangulation)
        tri = Delaunay(self._train_points)
        keep_mask = np.zeros(len(tri.simplices), dtype=bool)
        boundary_set: set = set()

        # 2. Filter triangles: keep only those with circumradius < 1/alpha
        #    Small circumradius = compact triangle = keep
        #    Large circumradius = stretched triangle = discard
        for idx, simplex in enumerate(tri.simplices):
            circumradius = _circumradius(self._train_points[simplex])
            if circumradius is None:
                continue
            if circumradius < 1.0 / self.alpha:
                keep_mask[idx] = True
                # 3. Track boundary edges: edges that belong to only ONE
                #    kept triangle are boundary edges (toggle in/out of set)
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    if edge in boundary_set:
                        boundary_set.remove(edge)
                    else:
                        boundary_set.add(edge)

        # Store kept simplices and boundary as coordinate pairs
        self._inside_simplices = tri.simplices[keep_mask]
        self._boundary_idx = list(boundary_set)
        self.boundary_edges = [(self._train_points[a], self._train_points[b])
                               for a, b in boundary_set]
        return self

    def query(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Test which points lie outside the alpha shape.

        Args:
            points: (M, 2) array of query points.

        Returns:
            Tuple of:
              - ``outside``: boolean array (M,), True if outside.
              - ``bdist``: float array (M,), distance to the nearest
                boundary edge (zero for inside points).
        """
        points = np.asarray(points, dtype=float)
        outside = np.ones(len(points), dtype=bool)

        if len(self._inside_simplices) == 0:
            return outside, np.full(len(points), np.inf)

        # Check each kept triangle: if a point's barycentric coords
        # are all >= 0, it's inside that triangle → mark as inside
        for simplex in self._inside_simplices:
            tri_verts = self._train_points[simplex]
            bary = _barycentric(tri_verts, points)
            outside[np.all(bary >= -1e-10, axis=1)] = False

        # For outside points, compute distance to nearest boundary edge
        bdist = np.zeros(len(points))
        outside_idx = np.where(outside)[0]
        if self._boundary_idx and len(outside_idx) > 0:
            edge_arr = np.array(self._boundary_idx)
            seg_a = self._train_points[edge_arr[:, 0]]
            seg_b = self._train_points[edge_arr[:, 1]]
            bdist[outside_idx] = _points_to_segments_min_dist(
                points[outside_idx], seg_a, seg_b)

        return outside, bdist

    def boundary_polygon(self) -> List[tuple]:
        """Walk the boundary edges to produce an ordered polygon.

        Connects the boundary edges into a continuous path by following
        neighbours. Works for a single connected component. Returns an
        empty list when the shape has no boundary.

        Returns:
            Ordered list of ``(x, y)`` tuples forming the polygon.
        """
        if not self.boundary_edges:
            return []

        # Build adjacency: each vertex maps to its neighbours
        edge_dict: dict = {}
        for seg in self.boundary_edges:
            start_pt = tuple(seg[0])
            end_pt = tuple(seg[1])
            edge_dict.setdefault(start_pt, []).append(end_pt)
            edge_dict.setdefault(end_pt, []).append(start_pt)

        # Walk the boundary starting from an arbitrary vertex
        start = next(iter(edge_dict))
        polygon = [start]
        visited = {start}
        current = start
        while True:
            nxt = None
            for neighbor in edge_dict[current]:
                if neighbor not in visited:
                    nxt = neighbor
                    break
            if nxt is None:
                break
            polygon.append(nxt)
            visited.add(nxt)
            current = nxt
        return polygon


def _circumradius(triangle: np.ndarray) -> float | None:
    """Circumradius of a 2-D triangle. Returns None for degenerate cases.

    The circumradius is the radius of the circle passing through all
    three vertices. Used to decide if a triangle is "compact" enough
    to keep in the alpha shape.

    Args:
        triangle: (3, 2) array of triangle vertices.

    Returns:
        Circumradius value, or None if the triangle has zero area.
    """
    side_a = np.linalg.norm(triangle[0] - triangle[1])
    side_b = np.linalg.norm(triangle[1] - triangle[2])
    side_c = np.linalg.norm(triangle[2] - triangle[0])
    semi = (side_a + side_b + side_c) / 2.0
    # Heron's formula for triangle area
    area = np.sqrt(
        max(semi * (semi - side_a) * (semi - side_b) * (semi - side_c), 0))
    if area == 0:
        return None
    return (side_a * side_b * side_c) / (4.0 * area)


def _barycentric(triangle: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of *points* w.r.t. *triangle*.

    Barycentric coords tell us where a point is relative to the triangle.
    If all three coords >= 0, the point is inside the triangle.

    Args:
        triangle: (3, 2) triangle vertices.
        points: (M, 2) query points.

    Returns:
        (M, 3) barycentric coordinate array.
    """
    vec_ca = triangle[2] - triangle[0]
    vec_ba = triangle[1] - triangle[0]
    vec_pa = points - triangle[0]
    dot00 = np.dot(vec_ca, vec_ca)
    dot01 = np.dot(vec_ca, vec_ba)
    dot11 = np.dot(vec_ba, vec_ba)
    dot02 = vec_pa @ vec_ca
    dot12 = vec_pa @ vec_ba
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-30)
    bary_u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    bary_v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return np.column_stack([1.0 - bary_u - bary_v, bary_v, bary_u])


def _points_to_segments_min_dist(points: np.ndarray, seg_a: np.ndarray,
                                 seg_b: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the nearest line segment.

    For each query point, finds the closest point on each boundary
    segment (clamped to the segment endpoints) and returns the
    minimum distance across all segments.

    Args:
        points: (P, 2) query points.
        seg_a: (E, 2) segment start points.
        seg_b: (E, 2) segment end points.

    Returns:
        (P,) array of minimum distances.
    """
    # Direction vector for each segment: shape (E, 2)
    vec_ab = seg_b - seg_a
    # Vector from each segment start to each query point: shape (P, E, 2)
    vec_ap = points[:, np.newaxis, :] - seg_a[np.newaxis, :, :]
    # Squared length of each segment: shape (E,)
    ab_dot = np.sum(vec_ab * vec_ab, axis=1)
    # Project each point onto each segment, clamp to [0, 1]
    proj = np.clip(
        np.sum(vec_ap * vec_ab[np.newaxis, :, :], axis=2) / (ab_dot + 1e-30), 0,
        1)
    # Closest point on each segment for each query point
    closest = (seg_a[np.newaxis, :, :] +
               proj[:, :, np.newaxis] * vec_ab[np.newaxis, :, :])
    # Distance from query point to closest point on each segment
    dists = np.linalg.norm(points[:, np.newaxis, :] - closest, axis=2)
    # Return minimum distance across all segments
    return dists.min(axis=1)
