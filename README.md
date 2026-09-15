# ai-trading-engine
AI Trading Engine v7.0 — Walk-Forward, Monte-Carlo, Kelly-Sizing


# AI Trading Engine v7.0

Eine Referenzimplementierung einer Machine-Learning-basierten Trading-Engine
mit **purged Walk-Forward-Validierung**, **Monte-Carlo-Robustheitstests** und
**Kelly-basiertem Positions-Sizing**.

> **Status:** Prototyp — Greenfield v7.0. Kein Migrationsaufwand von
> älteren Versionen, alle bekannten Fallstricke von Anfang an vermieden.

[![Tests](https://github.com/kptoerg-dev/ai-trading-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/kptoerg-dev/ai-trading-engine/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-yellow.svg)](LICENSE)

---

## Inhaltsverzeichnis

1. [Was ist das?](#was-ist-das)
2. [Warum dieses Projekt?](#warum-dieses-projekt)
3. [Die 7 Designprinzipien](#die-7-designprinzipien)
4. [Installation](#installation)
5. [Schnellstart](#schnellstart)
6. [Projektstruktur](#projektstruktur)
7. [Verwendung im Detail](#verwendung-im-detail)
8. [Konfiguration](#konfiguration)
9. [Tests](#tests)
10. [CI/CD](#cicd)
11. [Erweiterung](#erweiterung)
12. [Häufige Fragen](#häufige-fragen)
13. [Lizenz](#lizenz)

---

## Was ist das?

Eine **Trading-Engine** ist das Herzstück jedes systematischen Handelssystems.
Sie hat drei Aufgaben:

1. **Lernen:** Aus historischen Kursdaten erkennen, welche Muster
   profitabel sind.
2. **Simulieren:** Prüfen, ob die gelernten Muster auch auf ungesehenen
   Daten funktionieren (Backtest).
3. **Risiko steuern:** Bestimmen, wie viel Kapital pro Trade riskiert wird.

Diese Engine ist **kein fertiges Handelssystem**, sondern das **Framework**,
in das du dein eigenes Modell, deine eigenen Features und deine eigene
Strategie einhängst.

**Was drin ist:**

- Saubere Datenaufbereitung mit ATR-basierten Labels
- Zeitreihen-korrekte Kreuzvalidierung (purged Walk-Forward mit Embargo)
- Robuste Positionsgrößen-Berechnung (Kelly mit Unsicherheits-Penalty)
- Zwei Monte-Carlo-Tests für Robustheit
- Automatisierte Tests, die die Korrektheit dauerhaft sichern
- CI-Pipeline, die bei jedem Push prüft

**Was nicht drin ist:**

- Echte Broker-Anbindung (nur ein minimales Interface)
- Daten-Feeds (nutze deinen eigenen Loader)
- Fertiges ML-Modell (nutze deinen eigenen Classifier)
- Live-Trading-Loop

---

## Warum dieses Projekt?

Die meisten Trading-Backtests haben **denselben Fehler**: Sie sind zu
optimistisch. Das passiert aus zwei Gründen:

### 1. Data Leakage (Datenleckage)

Bei Zeitreihen ist das klassische k-Fold-Cross-Validation **falsch**.
Wenn du zufällig Trades in Trainings- und Test-Sets aufteilst, landen
Trades, die zeitlich nah beieinander liegen, in beiden Sets. Das Modell
"sieht" also beim Training indirekt die Zukunft.

**Lösung:** Purged Walk-Forward mit Embargo. Trainiere nur auf Vergangenheit,
teste auf Zukunft, und lasse eine Lücke (Embargo) zwischen beiden.

### 2. Label-Backtest-Mismatch

Wenn dein Modell auf Labels trainiert wird, die mit anderen Regeln erzeugt
wurden als dein Backtest verwendet, optimiert es auf das falsche Ziel.

**Beispiel:** Das Modell lernt "Take-Profit wurde am Entry-Tag getroffen →
Gewinn". Der Backtest ignoriert aber Trades, die nur den Entry-Tag treffen.
Ergebnis: Modell sagt 90 % Trefferquote, Backtest zeigt 40 %.

Diese Engine vermeidet beide Fallen **von der ersten Zeile an**.

---

## Die 7 Designprinzipien

Jedes Prinzip adressiert einen konkreten Fehler, der in Trading-Engines
häufig vorkommt.

### 1. Label-Fenster = Ausführungsfenster

**Das Problem:** Ein Trade wird bei Bar `i+1` (Open) eröffnet. Wenn
das Label aber bereits bei Bar `i+1` prüft, ob ein Stop-Loss oder
Take-Profit getroffen wurde, zählt dieser Bar als potenzieller Treffer.
Der Backtest schließt ihn aber aus (weil der Entry gerade erst
stattgefunden hat). Das Modell lernt Muster, die im Backtest nie
auftreten.

**Die Lösung:** SL/TP-Prüfung erst ab Bar `i+2`. In Code:

```python
for step in range(2, h + 2):   # Bar i+2 bis i+h+1
    ...
```

### 2. Explizite Klammern in Bool-Masken

**Das Problem:** In Python bindet `|` stärker als `>=`. Der Ausdruck

```python
np.arange(n) >= n - h - 1 | isna1 | isna2
```

wird geparst als

```python
np.arange(n) >= (n - h - 1 | isna1 | isna2)
```

— was völlig anderes bedeutet. NaN-Zeilen werden still durchgeschleust.

**Die Lösung:** Immer klammern:

```python
invalid = (
    (np.arange(n) >= n - h - 2)
    | out["LABEL_ENTRY"].isna().to_numpy()
    | out["ATR"].isna().to_numpy()
)
```

### 3. Ehrliche Monte-Carlo-Benennung

**Das Problem:** Ein MC-Test wird als "Ausführungs-Noise" bezeichnet,
aber die Formel `max(0, noise - 1)` macht jede Slippage einseitig
verschlechternd. Kein symmetrisches Rauschen.

**Die Lösung:** Die Funktion heißt ehrlich `mc_execution_stress` —
sie ist ein Stresstest, kein Rausch-Modell. Wer echte Unsicherheit
testen will, nutzt zusätzlich `mc_block_bootstrap`.

### 4. Pflichtfelder statt Wall-Clock-Zeitbomben

**Das Problem:** `Broker.exit()` setzte `exit_date` auf `pd.Timestamp.now()`.
Im Backtest wird das sofort überschrieben, aber in jedem anderen Kontext
(Notebook, Unit-Test, Live-Adapter) landet die aktuelle Systemzeit im
Trade-Log — ein stiller, schwer zu findender Bug.

**Die Lösung:** `ExitRequest.exit_date` ist Pflichtfeld. Kein Default.
Der Aufrufer muss wissen, welches Datum er schreiben will.

### 5. Parallelisierung auf der richtigen Ebene

**Das Problem:** Wenn Walk-Forward-Folds sequenziell laufen, aber
innerhalb jedes Folds `n_jobs=-1` gesetzt wird, entsteht ein Engpass:
Der Outer-Loop wartet auf jeden Fold, während die CPU zwischen den
Folds teilweise idle ist.

**Die Lösung:** Folds sind unabhängig — parallelisiere sie outer.
Innerhalb jedes Fits `n_jobs=1`. Am besten mit `joblib.Parallel`:

```python
Parallel(n_jobs=cfg.n_jobs_folds)(
    delayed(_run_single_fold)(f, data, cfg) for f in folds
)
```

### 6. Weicher Kelly-Floor

**Das Problem:** Ein harter Floor (`max(kelly, min_risk)`) springt bei
`raw ≈ min_probability` sofort auf die minimale Positionsgröße. Die
Positionsgröße ist dann nicht mehr proportional zur Edge.

**Die Lösung:** Optional weiche Rampe:

```python
w = clip((raw - min_prob) / ramp_width, 0, 1)
floor = w * min_risk
```

Bei kleinen Edges wächst die Position langsam mit. Bei großen Edges
greift der Floor nicht mehr.

### 7. Block-Bootstrap statt Trade-Permutation

**Das Problem:** Der Standard-MC-Test permutiert einzelne Trade-Returns
zufällig. Das testet Sequenzabhängigkeit, ignoriert aber
**Autokorrelation** — z. B. Verlustserien während eines Regime-Wechsels.

**Die Lösung:** Stationary Block-Bootstrap (Politis & Romano 1994).
Zieht zusammenhängende Blöcke statt einzelner Trades. Bildet
Regime-Cluster realistisch ab.

---

## Installation

### Voraussetzungen

- Python **3.10 oder neuer**
- `pip` (kommt mit Python)
- Optional: ein C-Compiler (falls `numba` genutzt werden soll)

### Standard-Installation

```bash
git clone https://github.com/kptoerg-dev/ai-trading-engine.git
cd ai-trading-engine
pip install -r requirements-dev.txt
```

Das installiert:
- `numpy` — numerische Operationen
- `pandas` — DataFrames
- `joblib` — Parallelisierung
- `pytest`, `hypothesis` — Tests
- `python-docx` — Word-Export (optional)

### Optionale Beschleunigung mit numba

Für große Datenmengen (100k+ Bars) lohnt sich `numba`:

```bash
pip install numba
```

**Wichtig:** Das Modul läuft auch **ohne** numba — nur langsamer.
Es fällt automatisch auf pure Python zurück.

### Auf Android (Termux)

```bash
pkg install python
pip install numpy pandas joblib
python trading_engine_v7.py
```

`numba` funktioniert auf Termux **oft nicht** (ARM-Kompilierung).
Das ist OK — der Code fällt automatisch auf pure Python zurück.

---

## Schnellstart

### Selbsttest ausführen

```bash
python trading_engine_v7.py
```

Das gibt aus:

```
============================================================
AI Trading Engine v7.0 — Self Test
============================================================
HAS_NUMBA  = True
HAS_PANDAS = True
HAS_JOBLIB = True

[Labeling]  valide LABEL_LONG-Zeilen: 477/500
[Labeling]  ✅ NaN-Propagation korrekt
[WF]        6 purged Folds erzeugt
[WF]        ✅ Folds disjunkt
[MC stress] p5=+0.00067  p50=+0.00098  prob_loss=41.35%
[MC block]  p5=+0.00021  p50=+0.00097  prob_loss=46.20%
[MC]        ✅ beide Tests laufen
[Broker]    TP @ 101.5
[Broker]    ✅ exit_date Pflichtfeld respektiert
[Kelly]     Rampe (soft floor):
            raw=0.50 -> risk_pct=0.00000
            raw=0.52 -> risk_pct=0.00000
            raw=0.53 -> risk_pct=0.00004
            raw=0.55 -> risk_pct=0.00120
            raw=0.60 -> risk_pct=0.00800
            raw=0.75 -> risk_pct=0.02000

============================================================
SELF TEST OK
============================================================
```

Wenn du das siehst, funktioniert alles.

### Erstes eigenes Beispiel

```python
from trading_engine_v7 import (
    Config, create_labels,
    build_purged_folds, run_walk_forward,
    mc_execution_stress, mc_block_bootstrap,
    kelly_size, Broker, ExitRequest,
    make_dummy_ohlcv,
)

# 1. Konfiguration
cfg = Config()

# 2. Daten (hier: synthetisch zum Testen)
df = make_dummy_ohlcv(500, seed=42)

# 3. Labels erzeugen
labeled = create_labels(df, cfg)
print(f"Valide Labels: {labeled['LABEL_LONG'].notna().sum()}/{len(df)}")

# 4. Walk-Forward-Folds vorbereiten
folds = build_purged_folds(len(df), cfg)
print(f"Folds: {len(folds)}")

# 5. Monte-Carlo-Robustheit
returns = labeled["LABEL_LONG"].dropna().to_numpy()
stress = mc_execution_stress(returns, cfg)
block = mc_block_bootstrap(returns, cfg)
print(f"Stress p5: {stress.p5:.4f}")
print(f"Block  p5: {block.p5:.4f}")

# 6. Kelly-Sizing für eine hypothetische Wahrscheinlichkeit
risk = kelly_size(raw=0.58, adjusted=0.005, cfg=cfg)
print(f"Risk: {risk:.4f}")
```

---

## Projektstruktur

```
ai-trading-engine/
│
├── trading_engine_v7.py        # Die Engine (Single-File)
├── test_v7_property.py         # Property-Based Tests
├── conftest.py                 # Test-Konfiguration (Hypothesis)
│
├── requirements.txt            # Laufzeit-Abhängigkeiten
├── requirements-dev.txt        # Zusätzliche Dev-Tools
├── pyproject.toml              # Build- und Test-Config
│
├── .github/
│   └── workflows/
│       └── tests.yml           # CI-Pipeline (GitHub Actions)
│
├── .gitignore                  # Ignorier-Regeln
├── LICENSE                     # MIT-Lizenz
├── CHANGELOG.md                # Versionshistorie
└── README.md                   # Diese Datei
```

### Was macht welche Datei?

| Datei | Zweck |
|---|---|
| `trading_engine_v7.py` | Das Herzstück — alle Funktionen |
| `test_v7_property.py` | Verifiziert, dass alles korrekt funktioniert |
| `conftest.py` | Steuert, wie intensiv die Tests laufen |
| `pyproject.toml` | Tool-Konfiguration (pytest, coverage) |
| `tests.yml` | CI-Pipeline für automatische Tests |
| `CHANGELOG.md` | Was hat sich wann geändert? |

---

## Verwendung im Detail

### 1. Daten vorbereiten

Die Engine erwartet einen `pandas.DataFrame` mit diesen Spalten:

| Spalte | Typ | Beschreibung |
|---|---|---|
| `open` | float | Eröffnungskurs pro Bar |
| `high` | float | Höchstkurs pro Bar |
| `low` | float | Tiefstkurs pro Bar |
| `close` | float | Schlusskurs pro Bar |
| `atr` | float | Average True Range (Volatilität) |

**Wichtig:** Der Index muss **zeitlich aufsteigend** sortiert sein.
Die Engine geht davon aus, dass Zeile `i+1` nach Zeile `i` kommt.

#### ATR berechnen

Falls deine Daten keine ATR-Spalte haben:

```python
import pandas as pd
import numpy as np

def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

df["atr"] = compute_atr(df, period=14)
```

#### Eigene Daten laden

```python
import pandas as pd

# Beispiel: CSV-Datei
df = pd.read_csv("btc_1h.csv", parse_dates=["timestamp"], index_col="timestamp")
df = df.rename(columns=str.lower)   # Spalten in Kleinbuchstaben
df = df.sort_index()                # zeitlich aufsteigend

# ATR berechnen, falls nicht vorhanden
if "atr" not in df.columns:
    df["atr"] = compute_atr(df)

# NaN-Zeilen am Anfang entfernen (durch rolling)
df = df.dropna(subset=["atr"]).reset_index(drop=True)
```

### 2. Labeling

```python
from trading_engine_v7 import Config, create_labels

cfg = Config(max_holding_bars=20, sl_atr_mult=1.5, tp_atr_mult=2.5)
labeled = create_labels(df, cfg)
```

Was passiert hier?

- Für jeden Bar `i` wird ein hypothetischer Trade eröffnet:
  - **Entry:** Bar `i+1` Open
  - **Stop-Loss:** Entry − `sl_atr_mult` × ATR
  - **Take-Profit:** Entry + `tp_atr_mult` × ATR
  - **Timeout:** Nach `max_holding_bars` Bars ohne SL/TP-Treffer
- Ergebnis pro Bar:
  - `+1.0` — Take-Profit getroffen
  - `-1.0` — Stop-Loss getroffen
  - `0.0` — Timeout (kein Treffer)
  - `NaN` — Bar ist invalide (fehlende Daten, Ende der Reihe)

Zusätzliche Spalten:

- `LABEL_ENTRY` — der Entry-Preis für Bar `i`
- `LABEL_ATR` — der ATR-Wert für Bar `i`
- `LABEL_LONG` — Label für Long-Trade
- `LABEL_SHORT` — Label für Short-Trade

### 3. Walk-Forward-Folds vorbereiten

```python
from trading_engine_v7 import build_purged_folds

folds = build_purged_folds(n=len(df), cfg=cfg)

for f in folds:
    print(f"Fold {f.fold_id}: "
          f"Train {f.train_idx[0]}-{f.train_idx[-1]}, "
          f"Test  {f.test_idx[0]}-{f.test_idx[-1]}")
```

Jeder Fold hat:
- `train_idx` — Indizes für Training (immer Vergangenheit)
- `test_idx` — Indizes für Test (immer Zukunft, nach Embargo)

**Wichtig:** Folds sind zeitlich disjunkt. Train endet **vor** Test
beginnt, mit Embargo-Lücke dazwischen.

### 4. Modell trainieren und evaluieren

Die Engine ist modell-agnostisch. Du lieferst drei Callbacks:

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

def fit_fn(train_df, cfg):
    """Trainiert ein Modell auf den Trainingsdaten eines Folds."""
    feature_cols = ["open", "high", "low", "close", "atr"]
    X = train_df[feature_cols].to_numpy()
    y = (train_df["LABEL_LONG"] > 0).astype(int)

    # Nur valide Labels nutzen
    mask = train_df["LABEL_LONG"].notna().to_numpy()

    model = ExtraTreesClassifier(
        n_estimators=300,
        n_jobs=cfg.n_jobs_trees,   # WICHTIG: 1 bei outer parallel!
        random_state=cfg.seed,
    )
    model.fit(X[mask], y[mask])
    return model


def predict_fn(model, test_df):
    """Liefert Wahrscheinlichkeiten für die Testdaten."""
    feature_cols = ["open", "high", "low", "close", "atr"]
    X = test_df[feature_cols].to_numpy()
    return model.predict_proba(X)[:, 1]


def eval_fn(preds, test_df, cfg):
    """Bewertet die Vorhersagen."""
    mask = test_df["LABEL_LONG"].notna().to_numpy()
    y_true = (test_df.loc[mask, "LABEL_LONG"] > 0).astype(int)
    if y_true.nunique() < 2:
        return {"auc": 0.5}
    return {"auc": roc_auc_score(y_true, preds[mask])}
```

Dann:

```python
from trading_engine_v7 import run_walk_forward

results = run_walk_forward(df, folds, cfg, fit_fn, predict_fn, eval_fn)

for r in results:
    print(f"Fold {r.fold_id}: AUC = {r.metrics['auc']:.4f}")

# Durchschnittliche AUC über alle Folds
avg_auc = sum(r.metrics["auc"] for r in results) / len(results)
print(f"\nDurchschnittliche AUC: {avg_auc:.4f}")
```

**Wichtig:** In `fit_fn` **musst** du `n_jobs=1` setzen. Die
Parallelisierung passiert outer (auf Fold-Ebene) via `joblib`.
Wenn du hier `n_jobs=-1` setzt, entsteht Oversubscription und
alles wird langsamer.

### 5. Monte-Carlo-Robustheit

Zwei Tests stehen zur Verfügung:

```python
from trading_engine_v7 import mc_execution_stress, mc_block_bootstrap

returns = labeled["LABEL_LONG"].dropna().to_numpy()

# Test 1: Slippage-Stresstest (einseitig)
stress = mc_execution_stress(returns, cfg)
print(f"Worst-Case-Erwartung (5% Perzentil): {stress.p5:.4f}")
print(f"Median:                              {stress.p50:.4f}")
print(f"Wahrscheinlichkeit Verlust:          {stress.prob_loss:.2%}")

# Test 2: Block-Bootstrap (Autokorrelation)
block = mc_block_bootstrap(returns, cfg)
print(f"Worst-Case (5% Perzentil):           {block.p5:.4f}")
print(f"Median:                              {block.p50:.4f}")
print(f"Wahrscheinlichkeit Verlust:          {block.prob_loss:.2%}")
```

**Was die Tests bedeuten:**

- `mc_execution_stress`: Simuliert systematisch schlechtere Ausführung.
  Antwortet auf: "Was, wenn meine Slippage schlimmer ist als erwartet?"
- `mc_block_bootstrap`: Simuliert alternative Historien mit ähnlicher
  Autokorrelation. Antwortet auf: "Wie stabil ist das Ergebnis, wenn
  Verlustserien anders verteilt gewesen wären?"

### 6. Kelly-Sizing

```python
from trading_engine_v7 import kelly_size

# Angenommen, dein Modell sagt 58% Wahrscheinlichkeit für einen Win
# und die Unsicherheits-Penalty reduziert auf 0.5
probability = 0.58
edge = probability * 2 - 1        # 0.16
adjusted = edge * 0.5             # 0.08 (nach Unsicherheits-Penalty)

risk = kelly_size(raw=edge, adjusted=adjusted, cfg=cfg)
print(f"Risk pro Trade: {risk:.4f} = {risk*100:.2f}% des Kapitals")
```

Ausgabe (mit `kelly_soft_floor=True`):

- Bei kleiner Edge: proportionale kleine Position
- Bei großer Edge: gedeckelt durch `max_risk_pct`

### 7. Broker-Anbindung (optional)

Der mitgelieferte `Broker` ist minimal:

```python
from datetime import datetime, timezone
from trading_engine_v7 import Broker, ExitRequest

broker = Broker(cash=10_000.0)

# Position schließen
trade = broker.exit(ExitRequest(
    price=101.5,
    exit_date=datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc),
    reason="TP",
))

print(trade)
# -> {'exit_price': 101.5, 'exit_date': ..., 'exit_reason': 'TP'}
```

**Wichtig:** `exit_date` ist Pflichtfeld. Kein Default. Das ist
Absicht — so kann keine Wall-Clock-Zeit versehentlich ins Log.

---

## Konfiguration

Alle Parameter sind in einem einzigen `Config`-Objekt:

```python
from trading_engine_v7 import Config

cfg = Config(
    # Reproduzierbarkeit
    seed=42,

    # Labeling
    max_holding_bars=20,      # Timeout nach N Bars
    sl_atr_mult=1.5,          # Stop-Loss = Entry - 1.5 * ATR
    tp_atr_mult=2.5,          # Take-Profit = Entry + 2.5 * ATR

    # Walk-Forward
    n_folds=6,                # Anzahl Folds
    embargo_bars=5,           # Lücke zwischen Train und Test
    n_jobs_folds=4,           # Parallelisierungsgrad (Folds)
    n_jobs_trees=1,           # Immer 1 bei outer parallel!

    # Kelly-Sizing
    min_probability=0.52,     # Ab wann wird getradet
    kelly_soft_floor=True,    # Weiche Rampe statt Hard-Floor
    kelly_ramp_width=0.05,    # Breite der Rampe
    min_risk_pct=0.002,       # 0.2% Mindest-Positionsgröße
    max_risk_pct=0.02,        # 2% Maximal-Positionsgröße

    # Monte-Carlo
    mc_n_paths=2000,          # Anzahl Simulationen
    mc_slippage_sigma=0.002,  # Slippage-Streuung
    mc_slippage_cap=1.5,      # Maximale Slippage
    mc_block_size=15,         # Blocklänge im Bootstrap
)
```

### Parameter-Erklärungen

#### `seed`

Zufalls-Seed für alle stochastischen Operationen (MC, Bootstrap,
Modelle). Fester Wert macht Ergebnisse reproduzierbar.

#### `max_holding_bars`

Nach wie vielen Bars ein Trade spätestens geschlossen wird, wenn
weder SL noch TP getroffen wurden. Größer = mehr Chancen auf TP,
aber auch mehr Risiko.

**Empfehlung:** 10–30, abhängig von deinem Zeitrahmen.

#### `sl_atr_mult`, `tp_atr_mult`

Stop-Loss und Take-Profit in ATR-Einheiten. `1.5` bedeutet: SL liegt
1.5 ATR unter dem Entry.

**Empfehlung:** `sl_atr_mult=1.0-2.0`, `tp_atr_mult=1.5-3.0`.
Ein TP/SL-Verhältnis über 1.5 ist meist sinnvoll.

#### `n_folds`

Wie viele Walk-Forward-Folds. Mehr Folds = robustere Schätzung,
aber weniger Trainingsdaten pro Fold.

**Empfehlung:** 5–10.

#### `embargo_bars`

Lücke zwischen Trainings- und Testdaten. Verhindert, dass Trades,
die im Training noch offen waren, in den Test "lecken".

**Empfehlung:** Mindestens `max_holding_bars`. Bei 20 Bars
Time-Out: `embargo_bars=20`.

#### `n_jobs_folds`

Wie viele Folds parallel laufen. Auf einem 8-Kern-Rechner: `8`.

#### `n_jobs_trees`

**Immer auf 1 lassen!** Das Modell wird innerhalb jedes Folds
seriell trainiert. Die Parallelisierung passiert auf Fold-Ebene.
Wenn du hier etwas anderes als 1 setzt, konkurrieren die Jobs
um CPU und alles wird langsamer.

#### `min_probability`

Ab welcher vorhergesagten Wahrscheinlichkeit ein Trade überhaupt
eröffnet wird. `0.52` bedeutet: erst ab 52% Siegwahrscheinlichkeit.

**Empfehlung:** 0.51–0.55. Zu niedrig → viele schlechte Trades.
Zu hoch → kaum Trades.

#### `kelly_soft_floor`

- `True`: weiche Rampe — kleine Edges bekommen kleine Positionen
- `False`: harter Floor — alles über `min_probability` bekommt
  mindestens `min_risk_pct`

**Empfehlung:** `True` für realistische Positionierung.

#### `mc_n_paths`

Wie viele Monte-Carlo-Simulationen. Mehr = stabilere Perzentile,
aber langsamer.

**Empfehlung:** 1000–5000.

#### `mc_slippage_sigma`

Streuung der Slippage-Verteilung. Höher = pessimistischeres
Stress-Szenario.

**Empfehlung:** 0.001–0.005 (0.1%–0.5% typische Slippage).

#### `mc_block_size`

Blocklänge im Bootstrap. Größer = mehr Autokorrelation wird
berücksichtigt.

**Empfehlung:** 10–30.

---

## Tests

### Alle Tests ausführen

```bash
pytest test_v7_property.py -v
```

### Was die Tests prüfen

Die Datei `test_v7_property.py` enthält **12 Tests**:

| Test | Prüft |
|---|---|
| `test_label_matches_backtest_long` | Label == Backtest (Long) |
| `test_label_matches_backtest_short` | Label == Backtest (Short) |
| `test_nan_in_atr_propagates_to_label` | NaN-Behandlung |
| `test_end_of_data_is_invalid` | Ende der Datenreihe |
| `test_mc_stress_is_pessimistic` | MC-Stress ist einseitig |
| `test_exit_request_requires_exit_date` | Pflichtfeld funktioniert |
| `test_broker_preserves_exit_date` | Datum wird nicht überschrieben |
| `test_purged_folds_are_disjoint` | Folds sind sauber getrennt |
| `test_kelly_zero_on_negative_raw` | Kelly bei negativer Edge |
| `test_kelly_soft_floor_monotonic` | Rampe ist monoton |
| `test_kelly_never_exceeds_max` | Max-Risk wird eingehalten |
| `test_block_bootstrap_detects_regimes` | Bootstrap erkennt Cluster |

### Was ist ein Property-Based Test?

Statt konkrete Beispiele zu prüfen ("Test A: erwartet X"), prüft
ein Property-Based Test **allgemeine Eigenschaften** über viele
zufällig generierte Eingaben.

Beispiel:

```python
@given(df=ohlcv_series())
def test_label_matches_backtest_long(df):
    ...
    assert abs(expected - actual) < 1e-6
```

Das bedeutet: "Für **jede** zufällige OHLCV-Reihe, die hypothesis
generiert, muss das Label exakt dem Backtest-Ergebnis entsprechen."

`hypothesis` generiert dabei **100 verschiedene Reihen** (konfigurierbar),
inklusive Edge-Cases wie:

- Extremwerte
- Leere/kleine Reihen
- NaN-Werte
- Große Spreads

Wenn ein Test fehlschlägt, zeigt hypothesis die **konkrete
Eingabe**, die das Problem auslöst. Das ist Gold wert für Debugging.

### Test-Intensität steuern

```bash
# Standard (schnell, 50 Beispiele)
pytest test_v7_property.py

# CI (gründlicher, 300 Beispiele)
HYPOTHESIS_PROFILE=ci pytest test_v7_property.py

# Fuzz (sehr gründlich, 3000 Beispiele)
HYPOTHESIS_PROFILE=fuzz pytest test_v7_property.py
```

### Coverage messen

```bash
pytest test_v7_property.py --cov=trading_engine_v7 --cov-report=term
```

Zeigt, welcher Prozentsatz des Codes durch Tests abgedeckt ist.

---

## CI/CD

Die Datei `.github/workflows/tests.yml` ist bereits konfiguriert.
Bei jedem Push auf `main`, `master` oder `develop` läuft:

1. **Smoke-Test** — `python trading_engine_v7.py`
2. **Property-Tests** — `pytest test_v7_property.py`
3. **Coverage-Report**

Und das auf **drei Python-Versionen** (3.10, 3.11, 3.12) parallel.

### Wo siehst du das Ergebnis?

1. GitHub-Repo öffnen
2. Tab **Actions** klicken
3. Den letzten Workflow-Lauf ansehen

Grüner Haken = alles OK. Roter X = etwas kaputt, klick für Details.

### Badge im README

Das Badge oben im README zeigt den **aktuellen Status**. Klick darauf
führt direkt zur Pipeline.

### Branch Protection (empfohlen)

Damit niemand kaputten Code nach `main` pusht:

1. Repo → **Settings** → **Branches**
2. **Add branch protection rule**
3. Branch name pattern: `main`
4. ✅ **Require status checks to pass before merging**
5. Wähle: `test (3.10)`, `test (3.11)`, `test (3.12)`

Jetzt kann niemand mehr mergen, wenn Tests rot sind.

---

## Erweiterung

### Eigenes Modell einhängen

Die Engine ist modell-agnostisch. Du lieferst drei Funktionen:

```python
def fit_fn(train_df, cfg):
    # Trainiere dein Modell hier
    # Rückgabe: dein trainiertes Modell-Objekt
    ...

def predict_fn(model, test_df):
    # Nutze dein Modell, um Wahrscheinlichkeiten zu liefern
    # Rückgabe: np.ndarray der Form (n_test,)
    ...

def eval_fn(preds, test_df, cfg):
    # Bewerte die Vorhersagen
    # Rückgabe: dict mit Metriken
    ...
```

Dann:

```python
results = run_walk_forward(df, folds, cfg, fit_fn, predict_fn, eval_fn)
```

**Beispiele für Modelle:**

- `sklearn.ensemble.ExtraTreesClassifier` (schnell, robust)
- `sklearn.ensemble.GradientBoostingClassifier`
- `xgboost.XGBClassifier`
- `lightgbm.LGBMClassifier`
- Eigene neuronale Netze (PyTorch, TensorFlow)

### Eigene Features

Der einfachste Weg: Füge Spalten zum DataFrame hinzu, **bevor** du
`create_labels` aufrufst. Die Engine nutzt sie dann automatisch,
wenn du sie in `fit_fn` referenzierst.

```python
df["rsi"] = compute_rsi(df["close"], period=14)
df["ema_fast"] = df["close"].ewm(span=10).mean()
df["ema_slow"] = df["close"].ewm(span=50).mean()
df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

# Dann in fit_fn:
feature_cols = ["open", "high", "low", "close", "atr",
                "rsi", "ema_diff"]
```

### Regimeabhängige Features

Ideen für Regime-Filter:

- **Trend-Regime:** ADX > 25
- **Volatilitäts-Regime:** ATR-Perzentil > 80
- **Session-Regime
