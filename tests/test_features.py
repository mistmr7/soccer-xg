"""Geometry sanity tests for the feature layer."""

import math

from soccerxg.features import freeze_frame_features, goal_angle, in_cone


def test_angle_shrinks_with_distance():
    near = goal_angle(114.0, 40.0)
    far = goal_angle(90.0, 40.0)
    assert near > far > 0


def test_angle_shrinks_off_center():
    central = goal_angle(108.0, 40.0)
    wide = goal_angle(108.0, 70.0)
    assert central > wide > 0


def test_penalty_spot_angle_value():
    a = goal_angle(108.0, 40.0)
    expected = 2 * math.atan(4.0 / 12.0)
    assert abs(a - expected) < 1e-9


def test_in_cone_keeper_on_line():
    assert in_cone(119.0, 40.0, 108.0, 40.0)


def test_out_of_cone_behind_shooter():
    assert not in_cone(100.0, 40.0, 108.0, 40.0)


def test_freeze_frame_parsing():
    ff = (
        '[{"location": [116.0, 40.0], "teammate": false, '
        '"position": {"name": "Goalkeeper"}},'
        '{"location": [110.0, 41.0], "teammate": false, '
        '"position": {"name": "Center Back"}},'
        '{"location": [109.0, 30.0], "teammate": true, '
        '"position": {"name": "Center Forward"}}]'
    )
    out = freeze_frame_features(ff, 100.0, 40.0)
    assert out["opp_in_cone"] == 1.0
    assert abs(out["nearest_opp"] - math.hypot(10.0, 1.0)) < 1e-9
    assert abs(out["gk_dist_goal"] - 4.0) < 1e-9


def test_freeze_frame_missing():
    out = freeze_frame_features(None, 100.0, 40.0)
    assert out["opp_in_cone"] != out["opp_in_cone"]
