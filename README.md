# bezier_with_matplotlib

Small demo that samples a cubic Bézier curve against obstacles, finds collision intervals, and visualizes the result with matplotlib. Obstacles can be defined analytically (rectangles/circles) or via a costmap-style grid, and several JSON map presets are included.

![Bezier obstacle-avoiding curve](docs/bezier_obstacle_curve.png)

## What it does
- Samples the curve, detects where it intersects obstacles, and reports the tau intervals (uniform or arc-length parameterization).
- Shows collision points, interval endpoints, and helper circles centered at segment midpoints with radius to the nearest obstacle contact.
- Prints the intervals and segment endpoints to stdout, and displays a plot of the scene.

## Run
```bash
pip install numpy matplotlib
python ver1.py
```
`ver1.py` loads the JSON maps in the repo root (e.g., `map_box_gauntlet.json`, `map_offset_diag.json`, `map_two_rects.json`, `map_wall_gap.json`, `map.json`). Update `map_path` in `demo()` if you want to swap maps or tweak parameters.
