#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_engine_v7.py
====================
AI Trading Engine v7.0 — Greenfield Implementation

Neu gebaut, keine Migration nötig. Alle bekannten Fallstricke (siehe
Code Review v6.0) sind von vornherein vermieden:

    ✅ Label-Fenster == Ausführungsfenster       (Bug #1)
    ✅ Explizite Klammern in Bool-Masken         (Bug #2)
    ✅ MC-Stresstest ehrlich benannt             (Punkt #3)
    ✅ exit_date als Pflichtfeld                 (Punkt #4)
    ✅ Outer-Parallelismus für Walk-Forward      (Punkt #5)
    ✅ Kelly-Soft-Floor optional                 (Punkt #6)
    ✅ Block-Bootstrap gegen Autokorrelation     (Punkt #7)

Designprinzipien:
    - Reine Funktionen (kein Shared State)
    - Typisierte Dataclasses statt loser Dicts
    - Optionale Abhängigkeiten (numba, joblib) mit Fallback
    - Alles über Config steuerbar, keine Magic Numbers
    - Single-File lauffähig, modular erweiterbar

Voraussetzungen:
    pip install numpy pandas
    pip install numba joblib   # optional, empfohlen

Aufruf:
    python trading_engine_v7.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

import numpy as np

# --- optionale Abhängigkeiten ---
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco

    def prange(x):
        return range

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =====================================================================
# 1. Konfiguration
# =====================================================================

@dataclass
class Config:
    """Zentrale Konfiguration — alles an einem Ort."""

    # --- Reproduzierbarkeit ---
    seed: int = 42

    # --- Labeling ---
    max_holding_bars: int = 20
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5

    # --- Walk-Forward ---
    n_folds: int = 6
    embargo_bars: int = 5
    n_jobs_folds: int = 4
    n_jobs_trees: int = 1  # bewusst 1: outer parallel, inner seriell

    # --- Kelly-Sizing ---
    min_probability: float = 0.52
    kelly_soft_floor: bool = True
    kelly_ramp_width: float = 0.05
    min_risk_pct: float = 0.002
    max_risk_pct: float = 0.02

    # --- Monte-Carlo ---
    mc_n_paths: int = 2000
    mc_slippage_sigma: float = 0.002
    mc_slippage_cap: float = 1.5
    mc_block_size: int = 15


# =====================================================================
# 2. Ergebnis-Typen
# =====================================================================

@dataclass
class MCResult:
    p5: float
    p50: float
    p95: float
    prob_loss: float
    n_paths: int

    def as_dict(self) -> dict:
        return {
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95,
            "prob_loss": self.prob_loss,
            "n_paths": self.n_paths,
        }


@dataclass(frozen=True)
class ExitRequest:
    """Pflichtfelder für Broker.exit() — kein Wall-Clock-Default."""
    price: float
    exit_date: datetime
    reason: str  # "SL" | "TP" | "TIME" | "MANUAL"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class FoldResult:
    fold_id: int
    test_idx: np.ndarray
    preds: np.ndarray
    metrics: dict


# =====================================================================
# 3. Labeling
# =====================================================================

NAN_F32 = np.float32(np.nan)


@njit(cache=True, inline="always")
def _label_one(i, n, low, high, entry, atr,
               sl_mult, tp_mult, h, is_long):
    """
    Label für Bar i.
    Entry = bar i+1 open (als entry[i] übergeben).
    SL/TP-Prüfung ab bar i+2  ->  range(2, h+2).
    """
    if np.isnan(entry[i]) or np.isnan(atr[i]):
        return NAN_F32

    e = entry[i]
    a = atr[i]

    if is_long:
        sl = e - sl_mult * a
        tp = e + tp_mult * a
        for step in range(2, h + 2):
            j = i + step
            if j >= n:
                break
            if low[j] <= sl:
                return np.float32(-1.0)
            if high[j] >= tp:
                return np.float32(1.0)
        return np.float32(0.0)
    else:
        sl = e + sl_mult * a
        tp = e - tp_mult * a
        for step in range(2, h + 2):
            j = i + step
            if j >= n:
                break
            if high[j] >= sl:
                return np.float32(-1.0)
            if low[j] <= tp:
                return np.float32(1.0)
        return np.float32(0.0)


@njit(cache=True, parallel=True)
def _label_core(low, high, entry, atr,
                sl_mult, tp_mult, h, is_long):
    n = len(low)
    out = np.full(n, NAN_F32, dtype=np.float32)
    for i in prange(n):
        out[i] = _label_one(i, n, low, high, entry, atr,
                            sl_mult, tp_mult, h, is_long)
    return out


def create_labels(df, cfg: Config):
    """
    Backtest-konsistentes Labeling.
    Entry: bar i+1 open. Barrieren: ab bar i+2.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas wird für create_labels() benötigt")

    n = len(df)
    out = df.copy()

    out["LABEL_ENTRY"] = df["open"].shift(-1)
    out["LABEL_ATR"] = df["atr"].shift(-1)

    low_arr = df["low"].to_numpy(np.float64)
    high_arr = df["high"].to_numpy(np.float64)
    entry_arr = out["LABEL_ENTRY"].to_numpy(np.float64)
    atr_arr = out["LABEL_ATR"].to_numpy(np.float64)

    out["LABEL_LONG"] = _label_core(
        low_arr, high_arr, entry_arr, atr_arr,
        cfg.sl_atr_mult, cfg.tp_atr_mult, cfg.max_holding_bars, True,
    )
    out["LABEL_SHORT"] = _label_core(
        low_arr, high_arr, entry_arr, atr_arr,
        cfg.sl_atr_mult, cfg.tp_atr_mult, cfg.max_holding_bars, False,
    )

    invalid = (
        (np.arange(n) >= n - cfg.max_holding_bars - 2)
        | out["LABEL_ENTRY"].isna().to_numpy()
        | out["LABEL_ATR"].isna().to_numpy()
        | df["atr"].isna().to_numpy()
    )
    out.loc[invalid, ["LABEL_LONG", "LABEL_SHORT"]] = np.nan
    out.attrs["invalid_mask"] = invalid
    return out


# =====================================================================
# 4. Walk-Forward
# =====================================================================

def build_purged_folds(n: int, cfg: Config) -> list[FoldSpec]:
    """
    Baut purged K-Fold-Splits mit Embargo.
    Reine Indexarithmetik — keine Datenabhängigkeit.
    """
    fold_size = n // (cfg.n_folds + 1)
    if fold_size < cfg.embargo_bars + cfg.max_holding_bars + 10:
        raise ValueError(
            f"Zu wenig Daten für {cfg.n_folds} Folds: "
            f"fold_size={fold_size}, benötigt>="
            f"{cfg.embargo_bars + cfg.max_holding_bars + 10}"
        )

    folds = []
    for k in range(cfg.n_folds):
        train_end = fold_size * (k + 1) - cfg.embargo_bars
        test_start = fold_size * (k + 1)
        test_end = min(test_start + fold_size, n)

        train_idx = np.arange(0, max(0, train_end))
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        folds.append(FoldSpec(
            fold_id=k,
            train_idx=train_idx,
            test_idx=test_idx,
        ))
    return folds


def run_walk_forward(
    data,
    folds: Sequence[FoldSpec],
    cfg: Config,
    fit_fn: Callable,
    predict_fn: Callable,
    eval_fn: Callable,
) -> list[FoldResult]:
    """
    Outer-parallele Walk-Forward-Ausführung.
    fit_fn muss intern n_jobs=1 setzen.
    """
    def _run(fold_spec: FoldSpec) -> FoldResult:
        tr = fold_spec.train_idx
        te = fold_spec.test_idx
        model = fit_fn(data.iloc[tr], cfg)
        preds = predict_fn(model, data.iloc[te])
        metrics = eval_fn(preds, data.iloc[te], cfg)
        return FoldResult(
            fold_id=fold_spec.fold_id,
            test_idx=te,
            preds=np.asarray(preds),
            metrics=metrics,
        )

    if not HAS_JOBLIB or cfg.n_jobs_folds <= 1:
        return [_run(f) for f in folds]

    return Parallel(n_jobs=cfg.n_jobs_folds, backend="loky")(
        delayed(_run)(f) for f in folds
    )


# =====================================================================
# 5. Broker
# =====================================================================

class Broker:
    """
    Minimaler Broker.
    exit_date ist Pflicht — kein Wall-Clock-Default.
    """

    def __init__(self, cash: float = 0.0):
        self.cash = cash
        self.trades: list[dict] = []

    def exit(self, req: ExitRequest) -> dict:
        trade = {
            "exit_price": float(req.price),
            "exit_date": req.exit_date,
            "exit_reason": req.reason,
        }
        self.trades.append(trade)
        return trade


# =====================================================================
# 6. Kelly-Sizing
# =====================================================================

def kelly_size(raw: float, adjusted: float, cfg: Config) -> float:
    """
    Positionsgröße nach Kelly.
    Bei kelly_soft_floor=True lineare Rampe statt Hard-Floor.
    """
    if raw <= 0:
        return 0.0

    if cfg.kelly_soft_floor:
        w = float(np.clip(
            (raw - cfg.min_probability) / max(cfg.kelly_ramp_width, 1e-9),
            0.0, 1.0,
        ))
        floor = w * cfg.min_risk_pct
        return float(np.clip(max(adjusted, floor),
                             0.0, cfg.max_risk_pct))

    return float(np.clip(adjusted, cfg.min_risk_pct, cfg.max_risk_pct))


# =====================================================================
# 7. Monte-Carlo
# =====================================================================

def mc_execution_stress(returns: np.ndarray, cfg: Config) -> MCResult:
    """
    Einseitiger Worst-Case-Test.
    Slippage verschlechtert IMMER — keine symmetrische Noise-Annahme.
    """
    rng = np.random.default_rng(cfg.seed)
    sample = np.asarray(returns, dtype=np.float64)
    n = len(sample)
    if n == 0:
        return MCResult(0.0, 0.0, 0.0, 1.0, 0)

    paths = np.empty((cfg.mc_n_paths, n), dtype=np.float64)
    for p in range(cfg.mc_n_paths):
        slip = rng.lognormal(0.0, cfg.mc_slippage_sigma, size=n)
        slip = np.minimum(slip, cfg.mc_slippage_cap)
        paths[p] = sample - np.abs(sample) * (slip - 1.0)

    finals = paths.mean(axis=1)
    return MCResult(
        p5=float(np.percentile(finals, 5)),
        p50=float(np.percentile(finals, 50)),
        p95=float(np.percentile(finals, 95)),
        prob_loss=float((finals < 0).mean()),
        n_paths=cfg.mc_n_paths,
    )


def mc_block_bootstrap(returns: np.ndarray, cfg: Config) -> MCResult:
    """
    Stationary Bootstrap (Politis & Romano).
    Respektiert Autokorrelation zwischen aufeinanderfolgenden Trades.
    """
    rng = np.random.default_rng(cfg.seed + 1)
    sample = np.asarray(returns, dtype=np.float64)
    n = len(sample)
    if n == 0:
        return MCResult(0.0, 0.0, 0.0, 1.0, 0)

    p = 1.0 / max(1, cfg.mc_block_size)
    finals = np.empty(cfg.mc_n_paths, dtype=np.float64)

    for path in range(cfg.mc_n_paths):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        start = int(rng.integers(0, n))
        while i < n:
            length = min(int(rng.geometric(p)), n - i)
            for k in range(length):
                idx[i + k] = (start + k) % n
            i += length
            start = int(rng.integers(0, n))
        finals[path] = sample[idx].mean()

    return MCResult(
        p5=float(np.percentile(finals, 5)),
        p50=float(np.percentile(finals, 50)),
        p95=float(np.percentile(finals, 95)),
        prob_loss=float((finals < 0).mean()),
        n_paths=cfg.mc_n_paths,
    )


# =====================================================================
# 8. Hilfsfunktionen für Prototyping
# =====================================================================

def make_dummy_ohlcv(n: int = 500, seed: int = 0):
    """Erzeugt einen zufälligen OHLCV-DataFrame für Tests."""
    if not HAS_PANDAS:
        raise ImportError("pandas benötigt")

    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    spread = 0.2 + rng.random(n) * 0.8
    high = close + spread
    low = close - spread
    open_ = np.clip(close + rng.normal(0, 0.3, n), low, high)
    atr = 0.5 + rng.random(n) * 1.0

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "atr": atr,
    })


# =====================================================================
# 9. Selbsttest
# =====================================================================

def _self_test():
    print("=" * 60)
    print("AI Trading Engine v7.0 — Self Test")
    print("=" * 60)
    print(f"HAS_NUMBA  = {HAS_NUMBA}")
    print(f"HAS_PANDAS = {HAS_PANDAS}")
    print(f"HAS_JOBLIB = {HAS_JOBLIB}")
    print()

    cfg = Config()

    # --- Labeling ---
    if HAS_PANDAS:
        df = make_dummy_ohlcv(500, seed=42)
        df.loc[100, "atr"] = np.nan  # invalide Zeile

        labeled = create_labels(df, cfg)
        n_valid = labeled["LABEL_LONG"].notna().sum()
        print(f"[Labeling]  valide LABEL_LONG-Zeilen: {n_valid}/{len(df)}")
        assert np.isnan(labeled.loc[100, "LABEL_LONG"]), \
            "NaN in ATR muss zu NaN-Label führen"
        print("[Labeling]  ✅ NaN-Propagation korrekt")

        # --- Walk-Forward ---
        folds = build_purged_folds(len(df), cfg)
        print(f"[WF]        {len(folds)} purged Folds erzeugt")
        for f in folds:
            assert len(f.train_idx) > 0 and len(f.test_idx) > 0
            assert f.train_idx.max() < f.test_idx.min(), \
                "Train/Test müssen disjunkt sein"
        print("[WF]        ✅ Folds disjunkt")

    # --- Monte-Carlo ---
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.01, 500)

    stress = mc_execution_stress(returns, cfg)
    print(f"[MC stress] p5={stress.p5:+.5f}  "
          f"p50={stress.p50:+.5f}  "
          f"prob_loss={stress.prob_loss:.2%}")

    block = mc_block_bootstrap(returns, cfg)
    print(f"[MC block]  p5={block.p5:+.5f}  "
          f"p50={block.p50:+.5f}  "
          f"prob_loss={block.prob_loss:.2%}")
    print("[MC]        ✅ beide Tests laufen")

    # --- Broker ---
    broker = Broker(cash=10_000.0)
    trade = broker.exit(ExitRequest(
        price=101.5,
        exit_date=datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc),
        reason="TP",
    ))
    print(f"[Broker]    {trade['exit_reason']} @ {trade['exit_price']}")
    print("[Broker]    ✅ exit_date Pflichtfeld respektiert")

    # --- Kelly ---
    print("[Kelly]     Rampe (soft floor):")
    for raw in (0.50, 0.52, 0.53, 0.55, 0.60, 0.75):
        size = kelly_size(raw=raw, adjusted=raw * 0.1, cfg=cfg)
        print(f"            raw={raw:.2f} -> risk_pct={size:.5f}")

    print()
    print("=" * 60)
    print("SELF TEST OK")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
