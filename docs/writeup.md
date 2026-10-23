# Building xG From Scratch, and Measuring What StatsBomb Knows That I Don't

*Draft for The Residual (an NBA newsletter playing an away fixture) and
the soccer-xg README. Every number traces to the repo:
github.com/mistmr7/soccer-xg.*

I've built the same model three times now, in three settings: a shot
probability model on NBA tracking data for a team's technical
assessment, the same architecture on 29 seasons of public NBA
play-by-play, and now expected goals on StatsBomb's open soccer data.
The architecture never changes: a ladder of models from a constant
baseline through zone lookups and logistic regression to gradient
boosting, scored on held-out data, calibration checked rather than
assumed. What changes is the information each dataset carries, and
measuring that difference has turned out to be the most interesting
output every time.

## The setup

73,463 non-penalty shots from StatsBomb's open data across ten
competitions (Messi-era La Liga, World Cups, Euros, the FA WSL, and
more), with entire tournaments held out for testing: the 2022 World
Cup and Euro 2024, 2,734 shots the models never see. Grouped
validation matters here; holding out random shots would leak match
and tournament context into the fit.

Features come in three layers. Geometry: distance to goal and the
angle subtended by the goal mouth. Context: body part, technique,
set-piece type, pressure and first-time flags. And the freeze frame,
StatsBomb's gift to the public: the position of every visible player
at the moment of the shot, from which I compute defenders between the
ball and the goal, the nearest defender's distance, and how far the
keeper is off his line. The freeze frame is the open data's version
of tracking data, and pricing it was the point of the exercise.

## The ladder

| Model | Log loss (held out) |
|---|---|
| Constant (base rate) | 0.3066 |
| Zone lookup | 0.2752 |
| Logistic regression | 0.2520 |
| Gradient boosting | 0.2517 |
| StatsBomb xG, same shots | 0.2468 |

Two things to see. First, the logistic nearly matches the gradient
booster, three tenths of a thousandth apart. In my NBA work the trees
earned real margins over linear models; here they find almost nothing
the linear model missed. Expected goals is very nearly an additive
physics problem: distance, angle, and who stands in the way. Second,
the from-scratch model lands 4.9 thousandths of log loss behind
StatsBomb's own xG, with an asymmetry worth stating plainly: my number
is strictly out-of-sample, entire tournaments held out, while
StatsBomb's published xG comes from a model that, as far as anyone
knows, trained on these very matches. I'm within five thousandths of a
benchmark that likely saw the answers.

## What the freeze frame is worth

Rerun the ladder with the three freeze-frame features removed and the
gradient booster falls from 0.2517 to 0.2582, and calibration error
grows by nearly a third. So knowing where the defenders and keeper
stand is worth 6.5 thousandths of log loss, which closes 57 percent of
the gap to StatsBomb. This is the third time I've priced information
this way: on NBA data, full tracking features were worth about 17
thousandths over public play-by-play. Where public data carries player
positions, the moat around proprietary models gets shallow.

## What the coefficients say

Standardized log-odds per one standard deviation, the full model:

```
dist        -1.117     gk_dist_goal  +0.275
header      -0.418     free_kick     +0.226
opp_in_cone -0.338     volley        -0.153
angle       +0.307     open_goal     +0.099
nearest_opp +0.305     one_on_one    +0.082
dist_angle  +0.290     under_pressure -0.024
```

Distance dominates everything, twice the weight of any other feature.
Headers carry a penalty nearly half a goal of log-odds per standard
deviation, soccer's version of the shot-type hierarchy. Defenders in
the shot cone is the third-strongest feature in the model.

Three details I'd flag to anyone building one of these. The
human-tagged "under pressure" flag is worth -0.093 when it's the only
pressure information available, and collapses to -0.024 once measured
defender geometry enters: the tag is a shadow of the measurement,
which is the whole argument for position-derived features over event
annotations in one coefficient. The first-time flag flips sign between
runs (positive without freeze frames, slightly negative with them),
classic selection: first-time shots happen in good positions, so
without defender context the flag proxies the situation; control for
the situation and the execution difficulty shows through. And the
keeper's distance off his line is strongly positive, which looks
backwards until you realize which shots those are: lobs, rounds, and
transition chances against a keeper who has already lost his goal.

## Does it transfer? Men's model, women's shots

One more experiment, because recruitment models live or die on whether
they generalize across populations: hold out every women's competition
in the corpus (the FA WSL, the Women's World Cups, the Women's Euros,
16,333 shots) and train on men's soccer alone.

The men's-trained model scores 0.2802 on women's shots against
StatsBomb's 0.2822 on the identical shots, and calibrates to 0.0091,
better than it calibrated on held-out men's tournaments. Two caveats
belong next to that sentence: the open data's StatsBomb xG values are
frozen at whatever model version shipped with each release, and their
women's xG may be less developed than their men's flagship. The
finding survives the caveats: in the zones where xG mass actually
lives, shot-conversion physics transfers across populations almost
perfectly. But the aggregate hides structure, so here is the
calibration by zone, men-trained model on women's shots:

Zones are radial, distance measured to the goal center, with "tight
angle" meaning the goal mouth subtends less than about 20 degrees:

| Zone | n | Observed | Predicted | StatsBomb |
|---|---|---|---|---|
| Long range (24+) | 4,072 | .036 | .026 | .024 |
| 18 to 24 out | 3,526 | .055 | .053 | .055 |
| Within 18, tight angle | 927 | .054 | .075 | .066 |
| Within 18, central | 7,070 | .149 | .150 | .146 |
| Within 6 | 738 | .396 | .390 | .369 |

The core is nearly perfect: close central shots price the same in
both games, and the raw base-rate difference between the populations
lives in which shots get taken, not what identical shots are worth.
The edges of the shot distribution hold two real deviations. Women's
long-range shots convert about 38 percent more often than
men's-trained geometry predicts (3.4 standard errors on 4,072 shots),
consistent with the keeper-coverage hypothesis: the goal is the same
size and the average keeper is not, and notably, StatsBomb's own xG
misses in the same direction. And women's close, tight-angle shots,
the near-post duels and byline chances, convert LESS than predicted
(2.8 SEs on 927 shots), which runs the wrong way for a pure
keeper-reach story; beating a keeper from a tight angle may be a
shot-velocity play, or within-zone shot selection may differ. Two
effects, opposite signs, both invisible to an aggregate calibration
number, which is the standing argument for reading the reliability
table instead of the summary statistic.

Then the symmetry check, because a real effect should mirror: train
on the 16,000 women's shots and test on the 60,000 men's. Both signs
flip on cue. Men's long-range shots convert BELOW what women's-trained
geometry expects (2.7 percent observed against 3.5 predicted, about
seven standard errors on 18,872 shots), and men's close tight-angle
shots convert ABOVE it (8.2 against 7.1, 2.5 SEs), with the quiet
zones staying quiet. The mechanism also shows up in the parameters
rather than just the residuals: fitted on the women's game, the
distance coefficient is about 20 percent shallower (-0.88 against
-1.12 log-odds per standard deviation), and the direct-corner
coefficient more than doubles, the same keeper-coverage physics
appearing in an unrelated shot type. One phenomenon, measured four
ways: forward residuals, mirrored residuals, a slope, and a corner.

A side observation from the same tables: the shipped StatsBomb values
under-predict six-yard conversion in both populations (.369 against
an observed .396 for women, .449 against .470 for men), each gap
modest alone but the same direction twice. It looks like model
compression at the distribution's edge, where regularization pulls
extreme probabilities toward the middle; this model sits on the
number in both tables.

A population-aware recalibration layer would absorb all of it and is
the obvious next arm.

## Why this exists

Partly because soccer was my first sport. Mostly because the method is
the point: build the ladder, hold out honestly, check calibration,
benchmark against the best available standard, and report the gap as a
measurement rather than an excuse. The model, data pipeline, tests,
and both run reports are public at github.com/mistmr7/soccer-xg. Data
provided by StatsBomb through their open-data program.
