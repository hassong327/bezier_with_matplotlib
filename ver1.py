import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 베지어 유틸
# ==========================
def de_casteljau(points, t):
    """
    De Casteljau 알고리즘으로 베지어 곡선 평가.
    points: (N+1, 2) 제어점
    t: [0,1] 스칼라
    return: (2,) 위치 벡터
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    for r in range(1, n):
        pts = (1 - t) * pts[:-1] + t * pts[1:]
    return pts[0]


def sample_bezier(points, num=200):
    """
    베지어 곡선을 균일한 tau 샘플로 평가.
    return: taus (num,), pts (num,2)
    """
    ts = np.linspace(0.0, 1.0, num)
    curve = np.array([de_casteljau(points, t) for t in ts])
    return ts, curve


def degree_elevate(points):
    """
    N차 베지어 제어점 -> (N+1)차 베지어 제어점
    PDF의 공식 그대로:
    Q0 = P0
    Q_{N+1} = P_N
    Q_j = j/(N+1) * P_{j-1} + (N+1-j)/(N+1) * P_j
    """
    P = np.array(points, dtype=float)
    N = len(P) - 1  # degree
    Q = np.zeros((N + 2, P.shape[1]), dtype=float)

    Q[0] = P[0]
    Q[-1] = P[-1]
    for j in range(1, N + 1):
        Q[j] = (j / (N + 1)) * P[j - 1] + ((N + 1 - j) / (N + 1)) * P[j]
    return Q


# ==========================
# 장애물: 폴리곤 + 충돌 체크
# ==========================
def point_in_polygon(pt, poly):
    """
    단순 ray-casting 기반 point-in-polygon.
    pt: (2,)
    poly: (M,2) 꼭짓점, 닫힌 다각형 (마지막=첫번째일 필요는 없음)
    """
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # y가 edge의 y범위 사이에 있을 때만 교차 검사
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_int > x:
                inside = not inside
    return inside


def is_point_in_any_obstacle(pt, obstacles):
    """
    obstacles: [poly1, poly2, ...], 각 poly는 (M,2) ndarray
    """
    return any(point_in_polygon(pt, poly) for poly in obstacles)


def nearest_obstacle_vertex(center, obstacles):
    """
    center로부터 가장 가까운 장애물 꼭짓점을 찾는다.
    """
    c = np.array(center, dtype=float)
    best_v = None
    best_d2 = None
    for poly in obstacles:
        for v in poly:
            d2 = np.sum((v - c) ** 2)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_v = v
    return np.array(best_v, dtype=float)


# ==========================
# 맵 로더 (JSON)
# ==========================
def load_map_from_json(path):
    """
    path에 있는 JSON을 읽어 start/end/obstacles 및 파라미터를 반환.
    obstacles: 각 항목은 {"type":"polygon","points":[[x,y],...]} 형태 사용.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start = np.array(data["start"], dtype=float)
    end = np.array(data["end"], dtype=float)

    obstacles = []
    for obs in data.get("obstacles", []):
        pts = np.array(obs.get("points", []), dtype=float)
        if len(pts):
            obstacles.append(pts)

    params = {
        "threshold": data.get("threshold", 0.5),
        "alpha": data.get("alpha", 0.2),
        "max_iter": data.get("max_iter", 100),
        "max_degree": data.get("max_degree", 100),
        "num_samples": data.get("num_samples", 400),
    }
    return start, end, obstacles, params


# ==========================
# 충돌 구간 (tau a,b) 찾기
# ==========================
def find_first_collision_interval(ctrl_points, obstacles, num_samples=300):
    """
    베지어 곡선을 샘플링하여, 처음으로 충돌이 시작되고 끝나는 tau 구간 (a,b)를 찾는다.
    - a: 충돌이 시작되는 샘플의 tau
    - b: 충돌이 끝나는 샘플의 tau
    """
    taus, curve = sample_bezier(ctrl_points, num_samples)
    inside = np.array([is_point_in_any_obstacle(p, obstacles) for p in curve])

    if not inside.any():
        return None  # 충돌 없음

    # True 구간들 중 '첫 번째 구간'의 시작/끝 index를 찾는다.
    idx = np.where(inside)[0]
    start_idx = idx[0]
    end_idx = start_idx
    for k in idx[1:]:
        if k == end_idx + 1:
            end_idx = k
        else:
            break  # 첫 번째 연속 구간만 사용

    a = taus[start_idx]
    b = taus[end_idx]
    return float(a), float(b)


# ==========================
# 메인 파이프라인
# ==========================
def generate_avoiding_bezier(
    start,
    end,
    obstacles,
    threshold=0.5,
    alpha=0.2,
    max_iter=100,
    max_degree=100,
    num_samples=400,
):
    """
    네가 제안한 파이프라인 구현:

    1. start-end를 잇는 직선 (1차 베지어)에서 시작
    2. 충돌하면 tau (a,b) 구하고, 그 구간의 중점으로 원 중심 설정
    3. 원 중심에서 threshold 거리 안에 제어점이 없으면 차수 승격 반복
    4. threshold 안에 있는 제어점 하나를 골라,
       - 원 밖 + alpha 만큼
       - (원 중심 -> 가장 가까운 장애물 vertex)의 반대 방향으로 이동
    이 과정을 충돌이 없어질 때까지, 혹은 max_iter / max_degree에 도달할 때까지 반복.
    """
    ctrl = np.array([start, end], dtype=float)  # degree 1

    for it in range(max_iter):
        interval = find_first_collision_interval(ctrl, obstacles, num_samples)
        if interval is None:
            print(f"[OK] collision-free after {it} iterations, degree={len(ctrl)-1}")
            return ctrl

        a, b = interval
        pa = de_casteljau(ctrl, a)
        pb = de_casteljau(ctrl, b)
        center = 0.5 * (pa + pb)

        # 3-1 / 3-2: threshold 안에 제어점이 있는지 확인
        dists = np.linalg.norm(ctrl - center, axis=1)
        cand_idx = np.where(dists <= threshold)[0]

        if cand_idx.size == 0:
            # threshold 안에 제어점이 없다면 차수 승격
            if (len(ctrl) - 1) >= max_degree:
                print(
                    f"[STOP] max degree {max_degree} reached but still colliding. "
                    "Return last curve."
                )
                return ctrl
            ctrl = degree_elevate(ctrl)
            # 차수 승격 후 다시 루프 (새로운 제어점들에 대해 3-1 검사)
            continue

        # threshold 안의 제어점 중에서 중심에 가장 가까운 것 하나 선택
        k = cand_idx[np.argmin(dists[cand_idx])]

        # 가장 가까운 장애물 vertex 찾기
        v_star = nearest_obstacle_vertex(center, obstacles)

        # (원 중심 -> vertex)의 '반대 방향'으로 control point를 민다 (장애물에서 멀어지도록)
        dir_vec = center - v_star
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            # vertex와 center가 거의 같으면, 충돌 구간의 법선 방향을 사용
            seg = pb - pa
            dir_vec = np.array([-seg[1], seg[0]])
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-8:
                # 이것도 0이면 그냥 x축 방향
                dir_vec = np.array([1.0, 0.0])
                norm = 1.0

        dir_unit = dir_vec / norm

        # 원 밖 + alpha 만큼: center + (threshold + alpha) * dir_unit
        new_point = center + (threshold + alpha) * dir_unit
        ctrl[k] = new_point

        # 디버그 로그
        print(
            f"[iter {it}] collision interval tau=({a:.3f},{b:.3f}), "
            f"move ctrl[{k}] to {new_point}"
        )

    print(
        f"[STOP] max_iter {max_iter} reached but still colliding. "
        "Return last curve."
    )
    return ctrl


# ==========================
# 데모용 메인
# ==========================
def demo():
    # JSON 맵 로드
    map_path = Path(__file__).with_name("map_box_gauntlet.json")
    start, end, obstacles, params = load_map_from_json(map_path)

    ctrl = generate_avoiding_bezier(
        start,
        end,
        obstacles,
        threshold=params["threshold"],  # 원 반지름
        alpha=params["alpha"],          # 여유 margin
        max_iter=params["max_iter"],
        max_degree=params["max_degree"],
        num_samples=params["num_samples"],
    )

    # 최종 곡선, 초기 직선 모두 그림
    ts, curve = sample_bezier(ctrl, num=400)
    ts0, curve0 = sample_bezier(np.array([start, end]), num=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    # 장애물
    for poly in obstacles:
        xs = np.append(poly[:, 0], poly[0, 0])
        ys = np.append(poly[:, 1], poly[0, 1])
        ax.fill(xs, ys, alpha=0.3, color="gray", label="obstacle")

    # 초기 직선
    ax.plot(curve0[:, 0], curve0[:, 1], "--", label="initial line")

    # 최종 베지어 곡선
    ax.plot(curve[:, 0], curve[:, 1], "-", label="final bezier")

    # 제어점 표시
    ax.plot(ctrl[:, 0], ctrl[:, 1], "o-", label="control points")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    ax.set_title("Bezier obstacle-avoiding curve")
    plt.show()


if __name__ == "__main__":
    demo()
