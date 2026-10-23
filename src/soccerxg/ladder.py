"""The model ladder, scored on held-out competitions, benchmarked
against StatsBomb's own xG on identical shots.

Rungs: constant make rate, zone lookup, logistic regression, gradient
boosting. Validation is grouped by competition-season: entire
tournaments are held out, so nothing about a test match, team, or
tournament context enters the fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import REPO, build_shots, design

OUT = REPO / "data" / "xg"
EPS = 1e-6

TEST_DEFAULT = [
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2024"),
]


def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_mae(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    rel = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")
    )
    w = rel.n / rel.n.sum()
    return float((rel.pred - rel.obs).abs().mul(w).sum())


FF_COLS = ["opp_in_cone", "nearest_opp", "gk_dist_goal", "ff_missing"]


def run(test_keys: list[tuple[str, str]], use_ff: bool = True,
        out_tag: str = "", exclude_zones: list[str] | None = None) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = build_shots()
    if exclude_zones:
        df = df[~df.zone.isin(exclude_zones)].copy()
        print(f"zones excluded from train AND test: {exclude_zones}")
    comp_only = {k[0] for k in test_keys if k[1] == "*"}
    full_keys = {k for k in test_keys if k[1] != "*"}
    key = list(zip(df.competition, df.season))
    is_test = pd.Series(
        [k in full_keys or k[0] in comp_only for k in key], index=df.index
    )
    train, test = df[~is_test], df[is_test]
    print(f"train {len(train):,} shots, test {len(test):,} shots "
          f"({sorted(set(zip(test.competition, test.season)))})")

    y_tr = train.goal.to_numpy()
    y_te = test.goal.to_numpy()
    report = {"n_train": len(train), "n_test": len(test)}

    report["ll_const"] = ll(y_te, np.full(len(test), y_tr.mean()))

    zr = train.groupby(train.zone).goal.mean()
    p_zone = test.zone.map(zr).fillna(y_tr.mean()).to_numpy()
    report["ll_zone"] = ll(y_te, p_zone)

    xtr, xte = design(train), design(test)
    if not use_ff:
        xtr = xtr.drop(columns=FF_COLS)
        xte = xte.drop(columns=FF_COLS)
        print("freeze-frame features EXCLUDED from this run")
    report["use_ff"] = use_ff
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(xtr, y_tr)
    report["ll_logistic"] = ll(y_te, lr.predict_proba(xte)[:, 1])

    coefs = lr.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    print("\nlogistic coefficients (log-odds per 1 SD, standardized):")
    for i in order:
        print(f"  {xtr.columns[i]:16s} {coefs[i]:+.3f}")
    report["coef"] = {xtr.columns[i]: float(coefs[i]) for i in order}
    print()

    hgb = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.08, max_iter=400,
        early_stopping=True, validation_fraction=0.1, random_state=7,
    )
    hgb.fit(xtr, y_tr)
    p_hgb = hgb.predict_proba(xte)[:, 1]
    report["ll_hgb"] = ll(y_te, p_hgb)
    report["calib_mae_hgb"] = calib_mae(y_te, p_hgb)

    sb_ok = test.shot_statsbomb_xg.notna().to_numpy()
    p_sb = test.shot_statsbomb_xg.to_numpy()[sb_ok]
    y_sb = y_te[sb_ok]
    report["ll_statsbomb"] = ll(y_sb, p_sb)
    report["ll_hgb_on_sb_rows"] = ll(y_sb, p_hgb[sb_ok])
    report["calib_mae_statsbomb"] = calib_mae(y_sb, p_sb)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if use_ff else "_noff"
    if comp_only:
        tag += "_transfer"
    if out_tag:
        tag += f"_{out_tag}"
    out = test[["competition", "season", "match_id", "player", "team",
                "goal", "dist", "angle", "zone"]].copy()
    out["xg_model"] = p_hgb
    out["xg_statsbomb"] = test.shot_statsbomb_xg.to_numpy()
    out.to_parquet(OUT / f"test_shots{tag}.parquet", index=False)
    (OUT / f"report{tag}.json").write_text(json.dumps(report, indent=1))

    print(
        f"const {report['ll_const']:.4f}  zone {report['ll_zone']:.4f}  "
        f"logistic {report['ll_logistic']:.4f}  hgb {report['ll_hgb']:.4f}  "
        f"calibMAE {report['calib_mae_hgb']:.4f}"
    )
    print(
        f"benchmark on identical shots: statsbomb {report['ll_statsbomb']:.4f}  "
        f"this model {report['ll_hgb_on_sb_rows']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="append", default=None,
                    help="held-out competition as 'Name:Season', repeatable")
    ap.add_argument("--no-ff", action="store_true",
                    help="exclude freeze-frame features (ablation)")
    ap.add_argument("--test-comp", action="append", default=None,
                    help="hold out an entire competition by name, all seasons")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. reverse")
    ap.add_argument("--exclude-zone", action="append", default=None,
                    help="drop a zone from train and test, repeatable")
    args = ap.parse_args()
    keys: list[tuple[str, str]] = []
    if args.test:
        keys += [tuple(t.split(":", 1)) for t in args.test]
    if args.test_comp:
        keys += [(c, "*") for c in args.test_comp]
    if not keys:
        keys = TEST_DEFAULT
    run(keys, use_ff=not args.no_ff, out_tag=args.tag,
        exclude_zones=args.exclude_zone)


if __name__ == "__main__":
    main()
