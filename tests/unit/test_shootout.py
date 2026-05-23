"""Unit tests for the shootout submodel."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wc2026.models.shootout import (
    ShootoutModel,
    fit_shootout_model,
    load_historical_shootouts,
    load_shootout_model,
    predict_shootout,
    simulate_shootout,
)

# --- ShootoutModel direct ---------------------------------------------------


def test_symmetric_prediction_at_equal_elo() -> None:
    m = ShootoutModel(slope=0.01, elo_lookup={"A": 1800.0, "B": 1800.0}, n_train=100)
    assert m.predict_home_win("A", "B") == pytest.approx(0.5)


def test_hand_computed_logistic_match() -> None:
    slope = 0.005
    elo_h, elo_a = 1900.0, 1800.0
    m = ShootoutModel(slope=slope, elo_lookup={"H": elo_h, "A": elo_a}, n_train=50)
    expected = 1.0 / (1.0 + math.exp(-slope * (elo_h - elo_a)))
    assert m.predict_home_win("H", "A") == pytest.approx(expected, abs=1e-12)


def test_missing_team_falls_back_to_half() -> None:
    m = ShootoutModel(slope=0.01, elo_lookup={"A": 1900.0}, n_train=10)
    assert m.predict_home_win("A", "Unknown") == 0.5
    assert m.predict_home_win("Unknown", "A") == 0.5


# --- fit_shootout_model -----------------------------------------------------


def _synth_shootouts(rng: np.random.Generator, n: int, true_slope: float, elo: dict[str, float]):
    teams = list(elo)
    rows = []
    for _ in range(n):
        i, j = rng.choice(len(teams), size=2, replace=False)
        home, away = teams[i], teams[j]
        p = 1.0 / (1.0 + math.exp(-true_slope * (elo[home] - elo[away])))
        winner = home if rng.random() < p else away
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": home,
                "away_team": away,
                "winner": winner,
            }
        )
    return pd.DataFrame(rows)


def test_fit_recovers_positive_slope_when_stronger_team_wins_more() -> None:
    rng = np.random.default_rng(7)
    elo = {f"T{i}": 1500.0 + 25.0 * i for i in range(8)}
    df = _synth_shootouts(rng, n=2000, true_slope=0.01, elo=elo)
    snap = pd.DataFrame({"team_name": list(elo), "rating": list(elo.values())})
    m = fit_shootout_model(df, snap)
    assert m.slope > 0.0
    # Roughly recovered (with L2 shrinkage and finite samples we don't insist on
    # tight numerical recovery — just direction + correct order of magnitude).
    assert 0.002 < m.slope < 0.02


def test_fit_drops_rows_with_unknown_teams() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 6),
            "home_team": ["A", "A", "B", "Unknown", "A", "B"],
            "away_team": ["B", "B", "A", "A", "B", "A"],
            "winner": ["A", "B", "A", "A", "A", "B"],
        }
    )
    snap = pd.DataFrame({"team_name": ["A", "B"], "rating": [1800.0, 1700.0]})
    m = fit_shootout_model(df, snap)
    assert m.n_train == 5  # the "Unknown" row dropped


def test_fit_raises_on_single_class_data() -> None:
    """All-home-wins (or all-home-losses) data has no logistic signal — surface
    a clear error rather than the cryptic sklearn message."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 6),
            "home_team": ["A", "A", "B", "A", "B", "A"],
            "away_team": ["B", "B", "A", "B", "A", "B"],
            "winner": ["A", "A", "B", "A", "B", "A"],  # home always wins
        }
    )
    snap = pd.DataFrame({"team_name": ["A", "B"], "rating": [1800.0, 1700.0]})
    with pytest.raises(ValueError, match="both classes"):
        fit_shootout_model(df, snap)


def test_fit_raises_on_missing_elo_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 6),
            "home_team": ["A"] * 6,
            "away_team": ["B"] * 6,
            "winner": ["A", "B", "A", "B", "A", "B"],
        }
    )
    bad_snap = pd.DataFrame({"team_name": ["A", "B"]})  # rating missing
    with pytest.raises(ValueError, match="elo snapshot missing columns"):
        fit_shootout_model(df, bad_snap)


def test_loader_raises_when_file_missing(tmp_path) -> None:
    from wc2026.models.shootout import load_historical_shootouts

    with pytest.raises(FileNotFoundError, match=r"shootouts\.csv"):
        load_historical_shootouts(target_dir=tmp_path)


def test_fit_raises_when_insufficient_overlap() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 2),
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "winner": ["A", "A"],
        }
    )
    snap = pd.DataFrame({"team_name": ["A", "B"], "rating": [1800.0, 1700.0]})
    with pytest.raises(ValueError, match="need >= 5"):
        fit_shootout_model(df, snap)


# --- 80/20 hold-out calibration smoke check ---------------------------------


def test_holdout_logloss_beats_coinflip_baseline() -> None:
    """On synthetic data generated from a known logistic, a fit on 80% must
    beat the 50/50 coin-flip log-loss on the held-out 20%."""
    rng = np.random.default_rng(123)
    elo = {f"T{i}": 1600.0 + 30.0 * i for i in range(10)}
    df = _synth_shootouts(rng, n=2000, true_slope=0.012, elo=elo)
    snap = pd.DataFrame({"team_name": list(elo), "rating": list(elo.values())})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    split = int(0.8 * len(df))
    train, test = df.iloc[:split], df.iloc[split:]
    m = fit_shootout_model(train, snap)
    p = test.apply(lambda r: m.predict_home_win(r["home_team"], r["away_team"]), axis=1)
    y = (test["winner"] == test["home_team"]).astype(int).to_numpy()
    eps = 1e-12
    ll_model = float(
        -np.mean(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps)))
    )
    ll_coin = -math.log(0.5)
    assert ll_model < ll_coin, f"model {ll_model:.4f} did not beat coin flip {ll_coin:.4f}"


# --- predict / simulate -----------------------------------------------------


def test_predict_shootout_returns_home_win_probability() -> None:
    m = ShootoutModel(slope=0.01, elo_lookup={"A": 1800.0, "B": 1700.0}, n_train=10)
    p = predict_shootout("A", "B", None, m)
    expected = 1.0 / (1.0 + math.exp(-0.01 * 100.0))
    assert p == pytest.approx(expected)


def test_simulate_shootout_distribution_matches_probability() -> None:
    rng = np.random.default_rng(2)
    m = ShootoutModel(slope=0.01, elo_lookup={"A": 2000.0, "B": 1700.0}, n_train=10)
    p = m.predict_home_win("A", "B")
    n = 5000
    wins_A = sum(simulate_shootout("A", "B", m, None, rng) == "A" for _ in range(n))
    # With n=5000 the 99% CI half-width on a binomial is ~1.5%; 4% is generous.
    assert abs(wins_A / n - p) < 0.04


# --- loader -----------------------------------------------------------------


def test_loader_reads_real_jurisoo_file_when_available() -> None:
    """Smoke check against the real on-disk shootouts.csv."""
    try:
        df = load_historical_shootouts()
    except FileNotFoundError:
        pytest.skip("jurisoo shootouts.csv not present locally")
    assert "home_team_won" in df.columns
    assert len(df) > 100
    assert df["date"].dtype.kind == "M"


# --- save / load round-trip --------------------------------------------------


def test_shootout_model_save_load_round_trip(tmp_path) -> None:
    """A round-tripped model must yield identical predictions."""
    original = ShootoutModel(
        slope=0.00123,
        elo_lookup={"Argentina": 2100.5, "France": 2050.25, "Atlantis": 1500.0},
        n_train=312,
    )
    path = tmp_path / "shootout" / "latest.json"
    original.save(path)
    assert path.exists()
    loaded = load_shootout_model(path)
    assert loaded.slope == pytest.approx(original.slope)
    assert loaded.n_train == original.n_train
    assert loaded.elo_lookup == original.elo_lookup
    # Equality of predictions for an Elo pair the model has.
    assert loaded.predict_home_win("Argentina", "France") == pytest.approx(
        original.predict_home_win("Argentina", "France")
    )


def test_shootout_model_save_creates_missing_parent_dirs(tmp_path) -> None:
    """save() must mkdir its parent (mirroring PoissonDC.save semantics)."""
    deep_path = tmp_path / "data" / "artifacts" / "shootout" / "latest.json"
    ShootoutModel(slope=0.001, elo_lookup={"A": 1500.0}, n_train=10).save(deep_path)
    assert deep_path.exists()
