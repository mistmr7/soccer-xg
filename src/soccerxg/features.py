"""Shot features from StatsBomb events.

Pitch coordinates are 120 by 80, attacking left to right, so the goal
line is x = 120 with posts at y = 36 and y = 44 (goal center 120, 40).

Three feature layers:
1. Geometry: distance to goal center and the angle subtended by the
   goal mouth (the angle between the vectors to the two posts), which
   captures distance and centrality in one physically meaningful
   number.
2. Context: body part, technique, set-piece type, pressure flags.
3. Freeze frame: positions of visible players at shot time. We derive
   the count of opponents inside the shot cone (the triangle from the
   shot location to the posts), the nearest opponent's distance, and
   the keeper's distance from the goal line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

GOAL = (120.0, 40.0)
POST_LOW = (120.0, 36.0)
POST_HIGH = (120.0, 44.0)


def goal_angle(x: float, y: float) -> float:
    """Angle in radians subtended by the goal mouth from (x, y)."""
    ax, ay = POST_LOW[0] - x, POST_LOW[1] - y
    bx, by = POST_HIGH[0] - x, POST_HIGH[1] - y
    dot = ax * bx + ay * by
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na == 0 or nb == 0:
        return 0.0
    c = max(-1.0, min(1.0, dot / (na * nb)))
    return math.acos(c)


def _sign(ox, oy, ax, ay, bx, by) -> float:
    return (ox - bx) * (ay - by) - (ax - bx) * (oy - by)


def in_cone(px: float, py: float, sx: float, sy: float) -> bool:
    """Is point (px, py) inside the triangle shot-post-post?"""
    d1 = _sign(px, py, sx, sy, *POST_LOW)
    d2 = _sign(px, py, *POST_LOW, *POST_HIGH)
    d3 = _sign(px, py, *POST_HIGH, sx, sy)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def freeze_frame_features(ff_json: str | None, sx: float, sy: float) -> dict:
    out = {"opp_in_cone": np.nan, "nearest_opp": np.nan, "gk_dist_goal": np.nan}
    if not ff_json:
        return out
    try:
        ff = json.loads(ff_json)
    except (TypeError, ValueError):
        return out
    opp_d = []
    cone = 0
    for p in ff:
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        px, py = float(loc[0]), float(loc[1])
        if p.get("teammate"):
            continue
        pos = (p.get("position") or {}).get("name", "")
        d = math.hypot(px - sx, py - sy)
        if pos == "Goalkeeper":
            out["gk_dist_goal"] = math.hypot(GOAL[0] - px, GOAL[1] - py)
            continue
        opp_d.append(d)
        if in_cone(px, py, sx, sy):
            cone += 1
    out["opp_in_cone"] = float(cone)
    out["nearest_opp"] = min(opp_d) if opp_d else np.nan
    return out


def zone(dist: pd.Series, angle: pd.Series) -> pd.Series:
    z = pd.Series("long", index=dist.index)
    z[(dist < 24)] = "edge"
    z[(dist < 18)] = "box_wide"
    z[(dist < 18) & (angle > 0.35)] = "box_central"
    z[dist < 6] = "six_yard"
    return z


def build_shots() -> pd.DataFrame:
    files = sorted(RAW.glob("shots_*.parquet"))
    if not files:
        raise SystemExit("no raw shot files; run soccerxg.pull first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df = df[df.location.notna()].copy()
    loc = df.location.map(lambda v: v if isinstance(v, (list, np.ndarray)) else None)
    df = df[loc.notna()].copy()
    df["x"] = df.location.map(lambda v: float(v[0]))
    df["y"] = df.location.map(lambda v: float(v[1]))

    df["goal"] = (df.shot_outcome == "Goal").astype(int)
    df["dist"] = np.hypot(GOAL[0] - df.x, GOAL[1] - df.y)
    df["angle"] = [goal_angle(x, y) for x, y in zip(df.x, df.y)]

    ff = [
        freeze_frame_features(f, x, y)
        for f, x, y in zip(df.shot_freeze_frame, df.x, df.y)
    ]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(ff)], axis=1)

    df["zone"] = zone(df.dist, df.angle)
    for flag in ["under_pressure", "shot_first_time", "shot_one_on_one", "shot_open_goal"]:
        df[flag] = df[flag].fillna(False).astype(bool).astype(int)

    df = df[df.shot_type != "Penalty"].copy()
    return df


def design(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["dist"] = df.dist.clip(0, 60)
    x["angle"] = df.angle
    x["dist_angle"] = x.dist * x.angle
    x["under_pressure"] = df.under_pressure
    x["first_time"] = df.shot_first_time
    x["one_on_one"] = df.shot_one_on_one
    x["open_goal"] = df.shot_open_goal
    x["header"] = (df.shot_body_part == "Head").astype(int)
    x["other_body"] = (df.shot_body_part == "Other").astype(int)
    x["volley"] = df.shot_technique.isin(["Volley", "Half Volley"]).astype(int)
    x["lob"] = (df.shot_technique == "Lob").astype(int)
    x["free_kick"] = (df.shot_type == "Free Kick").astype(int)
    x["corner_direct"] = (df.shot_type == "Corner").astype(int)
    x["opp_in_cone"] = df.opp_in_cone.fillna(df.opp_in_cone.median())
    x["nearest_opp"] = df.nearest_opp.fillna(df.nearest_opp.median()).clip(0, 20)
    x["gk_dist_goal"] = df.gk_dist_goal.fillna(df.gk_dist_goal.median()).clip(0, 30)
    x["ff_missing"] = df.opp_in_cone.isna().astype(int)
    return x
