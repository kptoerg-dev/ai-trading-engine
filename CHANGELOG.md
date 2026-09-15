# Changelog

Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).
Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

## [7.0.0] — 2026-09-15

Initiales Greenfield-Release. Alle bekannten Fallstricke aus der
v6.0-Review von Anfang an vermieden.

### Added

- Backtest-konsistentes Labeling (`create_labels`)
- Purged Walk-Forward-Folds (`build_purged_folds`)
- Outer-parallele Walk-Forward-Ausführung (`run_walk_forward`)
- Einseitiger Monte-Carlo-Stresstest (`mc_execution_stress`)
- Stationary Block-Bootstrap (`mc_block_bootstrap`)
- Kelly-Sizing mit weicher Rampe (`kelly_size`)
- Broker mit Pflicht-`exit_date` (`Broker`, `ExitRequest`)
- Property-Based Tests via `hypothesis`
- GitHub Actions CI (Python 3.10–3.12)

### Design Decisions

- **`numba` optional:** Fällt sauber auf pure Python zurück, damit
  Installation auf Android/Termux und allen CI-Umgebungen gelingt.
- **`joblib` optional:** Fällt auf serielle Schleife zurück.
- **Ein einziges `Config`-Objekt:** Keine drei separaten Configs.
- **Reine Funktionen:** Kein Shared State, testbar, picklable für
  Multiprocessing.
