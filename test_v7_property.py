#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v7_property.py
===================
Property-Based Tests für die v7.0 Trading Engine.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from trading_engine_v7 import (
    Config,
    create_labels,
    mc_execution_stress,
    mc_block_bootstrap,
    Broker,
    ExitRequest,
    kelly_size,
    build_purged_folds,
    make_dummy_ohlcv,
)


# ---------------------------------------------------------------------
# Referenz-Backtest (unabhängig, um Bug #1 zu verriegeln)
# ---------------------------------------------------------------------
def simulate_backtest_one_bar(low, high, entry, atr, i, h,
                              sl_mult, tp_mult, is_long):
    n = len(low)
    if np.isnan(entry) or np.isnan(atr):
        return np.nan

    if is_long:
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
        for step in range(2, h + 2):   # Entry-Tag i+1 übersprungen
            j = i + step
            if j >= n:
                break
            if low[j] <= sl:
                return -1.0
            if high[j] >= tp:
                return 1.0
        return 0.0
    else:
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr
        for step in range(2, h + 2):
            j = i + step
            if j >= n:
                break
            if high[j] >= sl:
                return -1.0
            if low[j] <= tp:
                return 1.0
        return 0.0


# ---------------------------------------------------------------------
# Hypothesis-Strategie: realistische OHLCV-Reihe
# ---------------------------------------------------------------------
@st.composite
def ohlcv_series(draw, min_bars=30, max_bars=80):
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))
    close = np.asarray(draw(st.lists(
        st.floats(min_value=50.0, max_value=200.0,
                  allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    )))
    spread = np.asarray(draw(st.lists(
        st.floats(min_value=0.1, max_value=2.0,
                  allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    )))
    high = close + spread
    low = close - spread
    open_ = np.clip(
        np.asarray(draw(st.lists(
            st.floats(min_value=50.0, max_value=200.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        ))),
        low, high,
    )
    atr = np.asarray(draw(st.lists(
        st.floats(min_value=0.05, max_value=3.0,
                  allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    )))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "atr": atr,
    })


# =====================================================================
# Bug #1 — Label == Backtest (long + short)
# =====================================================================
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(df=ohlcv_series())
def test_label_matches_backtest_long(df):
    cfg = Config(max_holding_bars=10, sl_atr_mult=1.0, tp_atr_mult=1.5)
    labeled = create_labels(df, cfg)

    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    entry = labeled["LABEL_ENTRY"].to_numpy(float)
    atr = labeled["LABEL_ATR"].to_numpy(float)
    labels = labeled["LABEL_LONG"].to_numpy(np.float32)

    n = len(df)
    last_valid = n - cfg.max_holding_bars - 2
    for i in range(last_valid):
        expected = simulate_backtest_one_bar(
            low, high, entry[i], atr[i], i,
            cfg.max_holding_bars, cfg.sl_atr_mult, cfg.tp_atr_mult,
            is_long=True,
        )
        actual = float(labels[i])
        if np.isnan(expected):
            assert np.isnan(actual), f"i={i}: label={actual}, exp=NaN"
        else:
            assert not np.isnan(actual) and abs(expected - actual) < 1e-6, \
                f"i={i}: label={actual}, exp={expected}"


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(df=ohlcv_series())
def test_label_matches_backtest_short(df):
    cfg = Config(max_holding_bars=10, sl_atr_mult=1.0, tp_atr_mult=1.5)
    labeled = create_labels(df, cfg)

    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    entry = labeled["LABEL_ENTRY"].to_numpy(float)
    atr = labeled["LABEL_ATR"].to_numpy(float)
    labels = labeled["LABEL_SHORT"].to_numpy(np.float32)

    n = len(df)
    last_valid = n - cfg.max_holding_bars - 2
    for i in range(last_valid):
        expected = simulate_backtest_one_bar(
            low, high, entry[i], atr[i], i,
            cfg.max_holding_bars, cfg.sl_atr_mult, cfg.tp_atr_mult,
            is_long=False,
        )
        actual = float(labels[i])
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert not np.isnan(actual) and abs(expected - actual) < 1e-6


# =====================================================================
# Bug #2 — NaN-Propagation in invaliden Zeilen
# =====================================================================
def test_nan_in_atr_propagates_to_label():
    df = make_dummy_ohlcv(200, seed=7)
    df.loc[42, "atr"] = np.nan
    labeled = create_labels(df, Config(max_holding_bars=10))

    assert np.isnan(labeled.loc[42, "LABEL_LONG"])
    assert np.isnan(labeled.loc[42, "LABEL_SHORT"])
    assert labeled.attrs["invalid_mask"][42]


def test_end_of_data_is_invalid():
    n = 60
    h = 10
    df = make_dummy_ohlcv(n, seed=3)
    labeled = create_labels(df, Config(max_holding_bars=h))

    mask = labeled.attrs["invalid_mask"]
    for i in range(n - h - 2, n):
        assert mask[i], f"Position {i} muss invalide sein"
    assert labeled["LABEL_LONG"].iloc[-h - 2:].isna().all()


# =====================================================================
# Punkt #3 — MC-Stresstest ist pessimistisch
# =====================================================================
@settings(max_examples=30, deadline=None)
@given(st.lists(
    st.floats(min_value=-0.05, max_value=0.05,
              allow_nan=False, allow_infinity=False),
    min_size=30, max_size=150,
))
def test_mc_stress_is_pessimistic(returns):
    sample = np.asarray(returns, dtype=np.float64)
    original_mean = sample.mean()
    if abs(original_mean) < 1e-9:
        return
    for seed in (0, 1, 2):
        cfg = Config(seed=seed, mc_n_paths=50)
        r = mc_execution_stress(sample, cfg)
        assert r.p50 <= original_mean + 1e-9


# =====================================================================
# Punkt #4 — Broker.exit() ohne Wall-Clock
# =====================================================================
def test_exit_request_requires_exit_date():
    with pytest.raises(TypeError):
        ExitRequest(price=100.0, reason="TP")


def test_broker_preserves_exit_date():
    broker = Broker()
    fixed = datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc)
    result = broker.exit(ExitRequest(
        price=101.5, exit_date=fixed, reason="TP",
    ))
    assert result["exit_date"] == fixed


# =====================================================================
# Punkt #5 — Purged Folds sind disjunkt
# =====================================================================
def test_purged_folds_are_disjoint():
    cfg = Config(n_folds=5, embargo_bars=5, max_holding_bars=10)
    folds = build_purged_folds(1000, cfg)
    assert len(folds) == 5
    for f in folds:
        assert len(f.train_idx) > 0
        assert len(f.test_idx) > 0
        assert f.train_idx.max() < f.test_idx.min(), \
            "Train und Test müssen zeitlich disjunkt sein"


# =====================================================================
# Punkt #6 — Kelly-Rampe
# =====================================================================
def test_kelly_zero_on_negative_raw():
    assert kelly_size(raw=-0.1, adjusted=0.5, cfg=Config()) == 0.0


def test_kelly_soft_floor_monotonic():
    cfg = Config(kelly_soft_floor=True,
                 min_probability=0.52,
                 kelly_ramp_width=0.05,
                 min_risk_pct=0.002,
                 max_risk_pct=0.02)
    raws = [0.51, 0.52, 0.53, 0.55, 0.57, 0.60, 0.70, 0.90]
    sizes = [kelly_size(raw=r, adjusted=r * 0.1, cfg=cfg) for r in raws]
    for a, b in zip(sizes, sizes[1:]):
        assert b >= a - 1e-12


def test_kelly_never_exceeds_max():
    cfg = Config()
    for raw in (0.5, 0.6, 0.9, 1.0):
        for adj in (0.001, 0.01, 0.05, 0.1):
            s = kelly_size(raw=raw, adjusted=adj, cfg=cfg)
            assert 0.0 <= s <= cfg.max_risk_pct + 1e-12


# =====================================================================
# Punkt #7 — Block-Bootstrap erkennt Autokorrelation
# =====================================================================
def test_block_bootstrap_detects_regimes():
    rng = np.random.default_rng(42)
    good = rng.normal(0.01, 0.005, 100)
    bad = rng.normal(-0.015, 0.005, 100)
    clustered = np.concatenate([good, bad])

    cfg = Config(seed=0, mc_n_paths=300, mc_block_size=20)
    block = mc_block_bootstrap(clustered, cfg)
    assert block.p5 < clustered.mean()
    assert block.prob_loss > 0.4


# =====================================================================
# Regression — konkretes Minimalbeispiel aus dem Review
# =====================================================================
def test_regression_entry_day_tp_touch():
    df = pd.DataFrame({
        "open":  [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "high":  [100.5, 105.0, 100.5, 100.5, 100.5, 100.5],
        "low":   [ 99.5,  99.5,  99.5,  99.5,  99.5,  99.5],
        "close": [100.0, 104.0, 100.0, 100.0, 100.0, 100.0],
        "atr":   [  1.0,   1.0,   1.0,   1.0,   1.0,   1.0],
    })
    cfg = Config(max_holding_bars=3, sl_atr_mult=1.0, tp_atr_mult=2.0)
    labeled = create_labels(df, cfg)
    assert labeled.loc[0, "LABEL_LONG"] == 0.0, \
        "Entry-Tag darf nicht als TP-Touch zählen (Bug #1)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
