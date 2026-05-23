# Methodology

This note summarises the modelling and evaluation pipeline behind the WC 2026
prediction platform. It covers the Stage 0 core (Poisson + Dixon-Coles with
weighted MLE, tournament simulator, backtest harness) and the Stage 1 model
enhancements (Elo-anchored prior, shootout submodel, isotonic recalibration).

## 1. Match model

We model each international fixture as an independent bivariate Poisson with
the Dixon-Coles low-score correction (Dixon & Coles 1997). Each team has a
latent attack and defence strength; expected goals are

    λ_home = exp(attack[home] + defence[away] + home_adv · (1 − neutral))
    λ_away = exp(attack[away] + defence[home])

Score probabilities factor as `Poisson(λ_home) · Poisson(λ_away)` with the
multiplicative τ correction on the four corner cells `(0-0, 0-1, 1-0, 1-1)`,
parameterised by ρ. Identifiability is fixed by `sum(attack) = sum(defence) = 0`
so the free parameter vector is `2·(N−1) + 2` for `N` teams. We fit by weighted
maximum likelihood with an analytic gradient under L-BFGS-B; the gradient is
unit-tested against `scipy.optimize.approx_fprime`.

## 2. Match weighting

Each match contributes weight `time_decay · importance`:

- **Exponential time decay** with a half-life selected by the Stage 0.6 sweep on
  the WC 2022 backtest. Ley, Van de Wiele & Van Eetvelde (2019) report a 390-day
  half-life for the EPL; international teams play ~10× fewer matches per year,
  so the optimum sits in the ~10-year range — our default is 3650 days.
- **Match importance** uses the eloratings.net K-factor schedule (K=60 for World
  Cup finals, 50 for continental finals, 40 for qualifiers/Nations Leagues, 30
  for other tournaments, 20 for friendlies). Source: eloratings.net "About"
  page, reviewed 2026-05-23.

## 3. Stage 1 enhancements (this PR)

### 3.1 Elo prior on attack/defence

`PoissonDCWithPrior` (in `src/wc2026/models/poisson_dc_with_prior.py`) adds an
L2 penalty pulling each team's `(attack, defence)` toward Elo-derived centres:

    strength      = (rating − mean_rating) / 100
    attack_centre = +strength
    defence_centre = −strength

The penalty is `λ · Σ ((a − a_c)² + (d − d_c)²)`. The analytic gradient
contribution is straightforward and is unit-tested against finite differences.
Teams missing from the Elo snapshot fall back to centre 0 (a no-op).

### 3.2 Penalty-shootout submodel

`fit_shootout_model` (in `src/wc2026/models/shootout.py`) fits a scikit-learn
no-intercept logistic regression on `[elo_home − elo_away]` predicting
`home_team_won`. `fit_intercept=False` enforces the symmetry `P(home wins) =
0.5` when Elo is equal. Trained on the full 585-row jurisoo shootout history
against the current Elo snapshot, the fitted slope is ~0.001, so a +100 Elo gap
maps to ~52% home-win probability — small but a strict improvement over the
50/50 coin-flip currently in `sim/knockout.py`. The submodel is **injectable**:
callers post-process a `KnockoutOutcome` with `decided_in == "shootout"` so we
do not have to modify `sim/knockout.py`.

### 3.3 Isotonic recalibration

`IsotonicCalibrator` fits one monotone non-parametric regression per outcome
(H, D, A) on the `(p_raw, indicator)` pairs from a hindcast, then re-normalises
the three calibrated probabilities to sum to 1 (Niculescu-Mizil & Caruana 2005).
`leave_one_out_recalibrate` reports honest LOO numbers (fit on N−1, apply to
held-out, repeat) so the calibrator never sees the row it is scoring.

## 4. Tournament simulator + tiebreakers

Group play and the 32-team knockout bracket are simulated by Monte Carlo on
the match model. Group rankings use the FIFA tiebreaker chain (points → goal
difference → goals scored → head-to-head → fair-play → drawing of lots), in
line with the academic backtests of Groll, Ley, Schauberger & Van Eetvelde
(2019). Knockout matches use regulation → extra time (rates `λ/3`) →
shootout (the new submodel above).

## 5. Backtest gates

Calibration is judged by negative log-loss, Brier score, and Ranked Probability
Score on a strict day-by-day hindcast — the model is refit only on matches
strictly before each prediction date (Gneiting & Raftery 2007;
Wheatcroft 2019). The hindcast harness is the platform's primary
"is-it-honest" gate.

## 6. Headline numbers

WC 2018 (64 matches) and WC 2022 (64 matches), with the production half-life
(3650 days) and 10-year history window:

| Variant | WC 2018 log-loss | WC 2022 log-loss |
|---|---|---|
| Baseline (PoissonDC) | **0.9585** | **1.0379** |
| + Elo prior (λ = 0.5) | 0.9572 | 1.0402 |
| + LOO isotonic | 1.5645 | 1.5431 |

References for comparison: uniform (1/3, 1/3, 1/3) log-loss = 1.0986; the
climatological log-loss on the WC 2022 base rates is ~1.05.

## 7. Limitations / open issues

- **Elo prior is a wash.** The prior gives a marginal log-loss improvement on
  WC 2018 (−0.0013) and a marginal degradation on WC 2022 (+0.0023). With a
  10-year history window the data already swamps the prior; we keep the code
  for future use (shorter half-lives, low-data team subsets) but the
  recommended production setting is `prior_strength = 0`. The current Elo
  snapshot is also post-2026, which introduces a soft look-ahead leak; for a
  strict comparison one would want an Elo snapshot dated at the hindcast
  cutoff.
- **LOO isotonic recalibration overfits on N=64.** Per-bin sample sizes on a
  single tournament are tiny, so a few [0.9, 1.0)-bin anomalies are enough to
  catastrophically inflate log-loss. The module is correct (synthetic tests
  pass) but should only be used when the calibration set covers many
  tournaments — recommended next step is to recalibrate on a rolling
  multi-tournament hindcast set.
- **Stage 0 limitations carry over.** Home-advantage is a single scalar (not
  per-team or per-confederation); the model assumes match outcomes are
  conditionally independent (no fatigue or rest effects); the score matrix is
  truncated at 10×10.
