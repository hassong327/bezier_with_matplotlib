import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class RectObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2


def bezier_point(ctrl: Sequence[Sequence[float]], t: float) -> np.ndarray:
    """Evaluate a Bézier curve at parameter t using De Casteljau."""
    ctrl = np.asarray(ctrl, dtype=float)
    n = len(ctrl) - 1
    points = ctrl.copy()
    for r in range(1, n + 1):
        points[: n - r + 1] = (1 - t) * points[: n - r + 1] + t * points[1 : n - r + 2]
    return points[0]


def world_to_grid(x: float, y: float, origin_x: float, origin_y: float, resolution: float) -> Tuple[int, int]:
    ix = int((x - origin_x) / resolution)
    iy = int((y - origin_y) / resolution)
    return ix, iy


def is_colliding_point_grid(
    x: float,
    y: float,
    grid: np.ndarray,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    resolution: float = 1.0,
) -> bool:
    ix, iy = world_to_grid(x, y, origin_x, origin_y, resolution)
    if ix < 0 or iy < 0 or ix >= grid.shape[1] or iy >= grid.shape[0]:
        return False
    return grid[iy, ix] != 0


def is_colliding_point_shapes(x: float, y: float, obstacles: Sequence[object]) -> bool:
    for obs in obstacles:
        if hasattr(obs, "contains") and obs.contains(x, y):
            return True
    return False


def find_collision_intervals(
    ctrl: Sequence[Sequence[float]],
    obstacles: Sequence[object],
    n_samples: int = 200,
    use_arc_length: bool = False,
    use_grid: bool = False,
    grid: np.ndarray | None = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    resolution: float = 1.0,
) -> Tuple[
    List[Tuple[float, float]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int]],
]:
    t_vals = np.linspace(0.0, 1.0, n_samples)
    pts = np.array([bezier_point(ctrl, t) for t in t_vals])

    collisions = []
    for x, y in pts:
        if use_grid and grid is not None:
            c = is_colliding_point_grid(x, y, grid, origin_x, origin_y, resolution)
        else:
            c = is_colliding_point_shapes(x, y, obstacles)
        collisions.append(c)
    collisions = np.array(collisions, dtype=bool)

    if use_arc_length:
        seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        tau_vals = cum_len / cum_len[-1] if cum_len[-1] > 0 else t_vals.copy()
    else:
        tau_vals = t_vals

    intervals: List[Tuple[float, float]] = []
    interval_indices: List[Tuple[int, int]] = []
    in_block = False
    start_idx = 0
    for i, flag in enumerate(collisions):
        if flag and not in_block:
            in_block = True
            start_idx = i
        if (not flag) and in_block:
            in_block = False
            end_idx = i - 1
            intervals.append((float(tau_vals[start_idx]), float(tau_vals[end_idx])))
            interval_indices.append((start_idx, end_idx))
    if in_block:
        intervals.append((float(tau_vals[start_idx]), float(tau_vals[-1])))
        interval_indices.append((start_idx, len(tau_vals) - 1))

    return intervals, t_vals, pts, collisions, tau_vals, interval_indices


def collision_segment_endpoints(
    ctrl: Sequence[Sequence[float]],
    t_vals: np.ndarray,
    interval_indices: List[Tuple[int, int]],
    intervals_tau: List[Tuple[float, float]],
) -> List[Tuple[Tuple[float, float], np.ndarray, np.ndarray]]:
    """Return tau interval with its start/end points; uses true t for evaluation."""
    segments: List[Tuple[Tuple[float, float], np.ndarray, np.ndarray]] = []
    for (idx_s, idx_e), (tau_s, tau_e) in zip(interval_indices, intervals_tau):
        t_s = float(t_vals[idx_s])
        t_e = float(t_vals[idx_e])
        p_start = bezier_point(ctrl, t_s)
        p_end = bezier_point(ctrl, t_e)
        segments.append(((tau_s, tau_e), p_start, p_end))
    return segments


def circle_radius_to_obstacles(center: np.ndarray, obstacles: Sequence[object]) -> float:
    """Compute smallest radius from center to first obstacle corner/boundary contact."""
    cx, cy = center
    radii = []
    for obs in obstacles:
        if isinstance(obs, RectObstacle):
            corners = [
                (obs.xmin, obs.ymin),
                (obs.xmin, obs.ymax),
                (obs.xmax, obs.ymin),
                (obs.xmax, obs.ymax),
            ]
            for vx, vy in corners:
                radii.append(float(np.hypot(vx - cx, vy - cy)))
        elif isinstance(obs, CircleObstacle):
            dist_center = np.hypot(obs.cx - cx, obs.cy - cy)
            radii.append(max(float(dist_center - obs.r), 0.0))
    if not radii:
        return 0.0
    return min(radii)


def demo_plot():
    # Sample obstacles
    obstacles = [
        RectObstacle(1.5, 3.0, 1.0, 3.5),
        RectObstacle(5.5, 7.0, 4.0, 5.5),
        CircleObstacle(6.0, 2.5, 0.9),
    ]

    # Optional grid (0=free,1=obstacle) showing how to mirror shapes onto a costmap-like array
    resolution = 0.1
    grid = np.zeros((120, 120), dtype=np.uint8)
    origin_x = 0.0
    origin_y = 0.0
    grid[10:35, 15:30] = 1
    grid[40:55, 55:70] = 1
    yy, xx = np.ogrid[:grid.shape[0], :grid.shape[1]]
    cx, cy, r = (6.0 - origin_x) / resolution, (2.5 - origin_y) / resolution, 0.9 / resolution
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    grid[circle_mask] = 1

    # Bézier control points
    ctrl = [(0.5, 0.5), (3.0, 6.0), (6.5, -1.0), (9.0, 6.0)]

    intervals, t_vals, pts, collisions, tau_vals, interval_indices = find_collision_intervals(
        ctrl,
        obstacles,
        n_samples=500,
        use_arc_length=True,
        use_grid=False,  # set True to use the grid map instead of analytic shapes
        grid=grid,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
    )
    segments = collision_segment_endpoints(ctrl, t_vals, interval_indices, intervals)
    circles = []
    for (_, p_start, p_end) in segments:
        mid = 0.5 * (p_start + p_end)
        r = circle_radius_to_obstacles(mid, obstacles)
        circles.append((mid, r))

    fig, ax = plt.subplots(figsize=(8, 6))
    for obs in obstacles:
        if isinstance(obs, RectObstacle):
            ax.add_patch(
                plt.Rectangle(
                    (obs.xmin, obs.ymin),
                    obs.xmax - obs.xmin,
                    obs.ymax - obs.ymin,
                    color="tab:red",
                    alpha=0.25,
                )
            )
        elif isinstance(obs, CircleObstacle):
            ax.add_patch(
                plt.Circle((obs.cx, obs.cy), obs.r, color="tab:red", alpha=0.25)
            )

    ax.plot(pts[:, 0], pts[:, 1], color="steelblue", label="Bézier curve")
    coll_pts = pts[collisions]
    if len(coll_pts):
        ax.scatter(coll_pts[:, 0], coll_pts[:, 1], color="orange", s=14, label="Collision")
        # Mark interval endpoints for clarity
        for idx, (_, p_start, p_end) in enumerate(segments):
            label = "Start/End" if idx == 0 else None  # avoid duplicate legend entries
            ax.scatter(p_start[0], p_start[1], color="red", s=30, marker="s", label=label)
            ax.scatter(p_end[0], p_end[1], color="red", s=30, marker="s")
        # Draw circles centered at segment midpoints with radius to first obstacle contact
        for idx, (mid, r) in enumerate(circles):
            label = "Mid circle" if idx == 0 else None
            circle = plt.Circle(mid, r, fill=False, color="black", linestyle="-", linewidth=1.5, label=label)
            ax.add_patch(circle)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Bézier collision intervals (tau)")

    print("Collision intervals (tau):", intervals)
    if segments:
        print("Collision segment endpoints (De Casteljau):")
        for i, ((tau_s, tau_e), p_s, p_e) in enumerate(segments, start=1):
            print(
                f"  Segment {i}: tau=({tau_s:.4f}, {tau_e:.4f}) "
                f"start=({p_s[0]:.3f}, {p_s[1]:.3f}) "
                f"end=({p_e[0]:.3f}, {p_e[1]:.3f})"
            )
        print("Midpoint circles (center, radius to first obstacle contact):")
        for i, (mid, r) in enumerate(circles, start=1):
            print(f"  Segment {i}: center=({mid[0]:.3f}, {mid[1]:.3f}), radius={r:.3f}")
    plt.show()


if __name__ == "__main__":
    demo_plot()
