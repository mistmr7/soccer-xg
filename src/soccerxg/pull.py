"""Pull shot events from StatsBomb open data, one parquet per
competition-season, with a manifest so reruns skip finished work.

statsbombpy fetches match by match from the open-data GitHub
repository, so a full pull takes a while; start it and walk away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
MANIFEST = RAW / "manifest.json"

COMPETITIONS = [
    "FIFA World Cup",
    "UEFA Euro",
    "La Liga",
    "Premier League",
    "1. Bundesliga",
    "Ligue 1",
    "Serie A",
    "FA Women's Super League",
    "UEFA Women's Euro",
    "Women's World Cup",
]

SHOT_COLS = [
    "id", "match_id", "minute", "period", "player", "team",
    "play_pattern", "location", "under_pressure",
    "shot_statsbomb_xg", "shot_outcome", "shot_body_part",
    "shot_technique", "shot_type", "shot_first_time",
    "shot_one_on_one", "shot_open_goal", "shot_freeze_frame",
]


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1))


def shots_for_match(sb, match_id: int) -> pd.DataFrame:
    ev = sb.events(match_id=match_id)
    shots = ev[ev["type"] == "Shot"].copy()
    shots = shots.reindex(columns=SHOT_COLS)
    shots["match_id"] = match_id
    return shots


def ff_to_json(ff) -> str | None:
    if isinstance(ff, list):
        return json.dumps(ff)
    return None


def pull(comps: list[str], max_matches: int | None) -> None:
    from statsbombpy import sb

    manifest = load_manifest()
    all_comps = sb.competitions()
    wanted = all_comps[all_comps.competition_name.isin(comps)]
    print(f"{len(wanted)} competition-seasons matched")

    for row in wanted.itertuples(index=False):
        key = f"{row.competition_id}_{row.season_id}"
        out = RAW / f"shots_{key}.parquet"
        if manifest.get(key) == "done" and out.exists():
            print(f"skip {row.competition_name} {row.season_name}")
            continue
        matches = sb.matches(
            competition_id=row.competition_id, season_id=row.season_id
        )
        ids = matches.match_id.tolist()
        if max_matches:
            ids = ids[:max_matches]
        parts = []
        for i, mid in enumerate(ids):
            try:
                parts.append(shots_for_match(sb, mid))
            except Exception as e:
                print(f"  match {mid} failed: {e}")
            if (i + 1) % 20 == 0:
                print(f"  {row.competition_name} {row.season_name}: "
                      f"{i + 1}/{len(ids)} matches")
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["shot_freeze_frame"] = df["shot_freeze_frame"].map(ff_to_json)
        df["competition"] = row.competition_name
        df["season"] = row.season_name
        RAW.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        if max_matches is None:
            manifest[key] = "done"
            save_manifest(manifest)
        print(f"wrote {out.name}: {len(df)} shots")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", action="append", default=None,
                    help="competition name filter, repeatable")
    ap.add_argument("--max-matches", type=int, default=None,
                    help="smoke-test cap per competition-season")
    args = ap.parse_args()
    pull(args.comp or COMPETITIONS, args.max_matches)


if __name__ == "__main__":
    main()
