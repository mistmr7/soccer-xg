# soccer-xg

An expected-goals model built from scratch on StatsBomb open data and
benchmarked against StatsBomb's own xG, the strongest public baseline
in the sport.

The architecture is a direct port of a shot-quality system built twice
before on NBA data (once on tracking data for an NBA team's technical
assessment, once on 29 seasons of public play-by-play): a ladder of
models from a constant baseline through zone lookups and logistic
regression to gradient boosting, every rung scored on held-out
competitions, with calibration checked rather than assumed.

Features come from three layers: shot geometry (distance and the angle
subtended by the goal mouth), context (body part, technique, set-piece
type, pressure, first-time, one-on-one), and the freeze frame, the
positions of every visible player at the moment of the shot, which
yields defender-in-cone counts, nearest-defender distance, and keeper
positioning, the soccer cousins of tracking-data contest features.

Validation is grouped by competition: train on older competitions,
test on entire held-out tournaments, so no match, team context, or
tournament leaks across the split.

## Run

```
uv run python -m soccerxg.pull
uv run python -m soccerxg.ladder
```

`pull` downloads shots competition by competition with a manifest, so
interrupting and rerunning skips finished work. `--max-matches N`
gives a quick smoke test. `ladder` builds features, fits the model
ladder, scores the held-out competitions, and benchmarks against
StatsBomb's shot_statsbomb_xg on identical shots.

## Data and attribution

Shot data from the [StatsBomb Open Data
repository](https://github.com/statsbomb/open-data), used under their
non-commercial license. StatsBomb require attribution: data provided
by StatsBomb. This project is independent research and is not
affiliated with StatsBomb.

## Author

Mike Mistarz. Related work: an NBA research program at
[nba-four-factors](https://github.com/mistmr7/nba-four-factors) and
writing at [The Residual NBA](https://theresidualnba.substack.com).
