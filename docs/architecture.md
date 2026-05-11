# Code Structure

What lives where + which script does what.

## Folder tree

```
smc-data-challenge/
├── data/                          # all on-disk data (gitignored except .gitkeep + gold)
│   ├── raw/spy_minute.parquet     # 1-min SPY bars from Alpaca
│   ├── processed/                 # H1 + splits + class weights
│   │   ├── spy_h1.parquet
│   │   ├── spy_h1_labeled.parquet # adds raw_label + encoded label columns
│   │   ├── spy_h1_{train,val,test}.parquet
│   │   └── class_weights.json
│   └── gold_labels.csv            # human-annotated validation set
│
├── src/                           # all importable code
│   ├── data/                      # acquisition, labelling, splitting, windowing
│   │   ├── download.py            # Alpaca SDK → raw parquet
│   │   ├── process.py             # 1-min → H1, RTH filter, integrity checks
│   │   ├── normalize.py           # per-window min-max (used live + offline)
│   │   ├── split.py               # temporal split, no shuffle
│   │   ├── window.py              # 60-bar sliding windows + session gap helper
│   │   ├── pipeline.py            # build_pipeline orchestrator
│   │   ├── annotate.py            # gold-set sampler + Plotly window renderer
│   │   └── labels/                # labeller package
│   │       ├── base.py            # BaseLabeller ABC + LABELLERS registry
│   │       ├── fvg.py             # FVGLabeller — raw 3-candle geometry @ N+1
│   │       ├── valid_fvg.py       # ValidFVGLabeller — 6-criteria SMC @ N+2
│   │       ├── bos.py             # Break-of-Structure detector (used by valid_fvg)
│   │       └── sr.py              # Pivot S/R detector (used by valid_fvg)
│   │
│   ├── features/                  # feature engineering for non-DL
│   │   └── window_features.py     # 60×5 window → feature vector for XGBoost
│   │
│   ├── models/                    # model architectures
│   │   ├── lstm.py                # FVGLSTMClassifier (2-layer unidir, MPS-safe)
│   │   └── xgboost_baseline.py    # GBM hyperparameter defaults
│   │
│   ├── training/                  # loss + optim + early stop
│   │   ├── loss.py                # WeightedCE, FocalLoss
│   │   ├── early_stop.py          # EMA-smoothed patience
│   │   └── train_utils.py         # seeding, device pick
│   │
│   ├── inspect/                   # offline model inspection toolkit
│   │   ├── base.py                # ModelAdapter ABC (shared with src/live)
│   │   ├── registry.py            # pkg-walks adapters/ and registers
│   │   ├── runner.py              # multi-adapter inference over windows
│   │   ├── stats.py               # F1, confusion, agreement
│   │   ├── outcomes.py            # TP/SL/undecided walk over future bars
│   │   ├── viz.py                 # Plotly per-window + per-model timeline
│   │   ├── report.py              # writes summary.md + HTML plots
│   │   └── adapters/              # drop a file here = new model auto-discovered
│   │       ├── lstm_adapter.py
│   │       ├── xgboost_adapter.py
│   │       └── _xgb_worker.py     # subprocess for Python 3.14+ segfault workaround
│   │
│   └── live/                      # live paper-trading harness
│       ├── stream.py              # AlpacaBarStream (WebSocket 1-min)
│       ├── window_builder.py      # 1-min → RTH-anchored H1, emits 60-bar windows
│       ├── decision.py            # SingleModelDecision (threshold + gap extraction)
│       ├── execution.py           # PaperExecutor (bracket orders + safety guards)
│       ├── slippage.py            # 1-tick overlay for honest P&L
│       ├── logger.py              # parquet bars + sqlite trades + jsonl events
│       └── replay.py              # deterministic session replay for debug
│
├── scripts/                       # CLI entry points
│   ├── annotate_gold_set.py       # interactive Plotly gold-set annotation
│   ├── count_valid_fvg.py         # sparsity gate for ValidFVGLabeller tuning
│   ├── persist_labels.py          # refresh spy_h1_labeled.parquet
│   ├── train_lstm.py              # train + save LSTM
│   ├── train_xgboost.py           # train + save XGBoost
│   ├── inspect_models.py          # offline inspector CLI
│   └── paper_trade.py             # live paper trader CLI
│
├── notebooks/                     # CRISP-DM presentation notebooks
│   ├── 01-data-understanding.ipynb
│   └── 02-gold-annotation.ipynb
│
├── tests/                         # pytest — mirrors src/ layout
│   ├── data/                      # pipeline, labels, windowing
│   ├── inspect/                   # adapter, registry, runner, stats
│   └── live/                      # stream, window builder, decision, executor
│
├── checkpoints/                   # trained weights (gitignored)
│   ├── lstm/lstm_seed42.pt
│   └── xgboost/xgb_seed42.ubj
│
├── logs/                          # paper-trading session logs (gitignored)
│   └── paper/<session-id>/{bars.parquet, events.jsonl, trades.sqlite}
│
├── reports/                       # inspector HTML output (gitignored)
│   └── inspect/<timestamp>/{summary.md, plots/}
│
├── docs/                          # this directory
├── .nb-suite/                     # research/plan/build logs
└── CLAUDE.md                      # project rules for AI assistance
```

## High-level component map

```mermaid
flowchart LR
    subgraph DATA["src/data/"]
        DL[download.py]
        PR[process.py]
        LB["labels/<br/>FVG, ValidFVG, BOS, SR"]
        SP[split.py]
        WD[window.py]
        NM[normalize.py]
        PL[pipeline.py]
    end

    subgraph FEAT["src/features/"]
        WF[window_features.py]
    end

    subgraph MOD["src/models/"]
        LS[lstm.py]
        XB[xgboost_baseline.py]
    end

    subgraph TR["src/training/"]
        LO[loss.py]
        ES[early_stop.py]
    end

    subgraph INS["src/inspect/"]
        BA[base.py — ModelAdapter ABC]
        RG[registry.py]
        RN[runner.py]
        ST[stats.py]
        OC[outcomes.py]
        VZ[viz.py]
        AD["adapters/<br/>lstm + xgboost"]
    end

    subgraph LIV["src/live/"]
        SM[stream.py]
        WB[window_builder.py]
        DC[decision.py]
        EX[execution.py]
        LG[logger.py]
        RP[replay.py]
    end

    DL --> PR --> LB --> SP --> WD --> PL
    NM --> WD
    NM --> WB
    WD --> WF
    WF --> XB
    WD --> LS
    LO --> LS
    LO --> XB
    ES --> LS
    BA --> AD
    AD --> RN
    RN --> ST --> VZ
    OC --> VZ
    BA --> DC
    SM --> WB --> DC --> EX --> LG
    LG --> RP
```

## Offline data flow (training + inspection)

```mermaid
sequenceDiagram
    participant U as scripts/
    participant DL as download
    participant PR as process
    participant LB as labeller
    participant SP as split
    participant WD as window
    participant TR as trainer
    participant CK as checkpoints/
    participant IN as inspector

    U->>DL: download_spy_h1()
    DL-->>U: data/raw/spy_minute.parquet
    U->>PR: 1-min → H1 (RTH-anchored)
    PR-->>U: data/processed/spy_h1.parquet
    U->>LB: label(df)  [FVG or ValidFVG]
    LB-->>U: raw_label + encoded label
    U->>SP: temporal_split (no shuffle)
    SP-->>U: train / val / test parquets
    U->>WD: 60-bar sliding windows
    WD-->>TR: (N, 60, 5) + labels
    TR-->>CK: lstm.pt / xgb.ubj
    CK-->>IN: load via adapter
    IN-->>IN: predict + outcome sim + plots
```

## Live data flow (paper trading)

```mermaid
sequenceDiagram
    participant ALP as Alpaca WS
    participant ST as stream.py
    participant WB as window_builder.py
    participant DC as decision.py
    participant EX as execution.py
    participant LG as logger.py
    participant ALP2 as Alpaca paper

    Note over WB: REST gap-fill on startup<br/>(60-bar warmup)
    ALP-->>ST: 1-min SPY bar
    ST->>WB: bar
    WB->>WB: buffer + resample H1<br/>(closed='left' label='left' offset='30min')
    Note over WB: emit on each H1 close
    WB->>DC: WindowEvent (60, 5)
    DC->>DC: adapter.predict_proba<br/>+ threshold gate<br/>+ gap extraction
    DC->>EX: TradeAction (entry, SL, TP)
    EX->>EX: 6 safety guards<br/>(DD, blackout, sizing, ...)
    EX->>ALP2: bracket order
    ALP2-->>EX: fills
    EX->>LG: order + fills + equity
    LG->>LG: parquet + sqlite + jsonl
```

## Scripts — what each does

| Script | Purpose | Typical command |
|--------|---------|-----------------|
| `scripts/train_xgboost.py` | Train XGBoost on windowed features | `python scripts/train_xgboost.py` |
| `scripts/train_lstm.py` | Train LSTM (MPS or CPU) | `python scripts/train_lstm.py` |
| `scripts/persist_labels.py` | Refresh `spy_h1_labeled.parquet` after labeller config change | `python scripts/persist_labels.py` |
| `scripts/count_valid_fvg.py` | Sparsity gate — count positives produced by current `ValidFVGLabeller` config | `python scripts/count_valid_fvg.py` |
| `scripts/annotate_gold_set.py` | Interactive Plotly annotation of gold validation set | `python scripts/annotate_gold_set.py` |
| `scripts/inspect_models.py` | Offline model inspection on test set with TP/SL outcome simulation | `python scripts/inspect_models.py --models lstm xgboost --lookahead-bars 20` |
| `scripts/paper_trade.py` | Live paper trading via Alpaca (one process per model) | `python scripts/paper_trade.py --model lstm:checkpoints/lstm/lstm_seed42.pt --session lstm-001` |

## Tools / external services

| Tool | Where used | Purpose |
|------|------------|---------|
| `alpaca-py` | `src/data/download.py`, `src/live/stream.py`, `src/live/execution.py` | Historical 1-min bars + live WebSocket + paper bracket orders |
| `exchange_calendars` | `src/data/process.py`, `src/live/window_builder.py` | NYSE schedule, half-day + holiday detection |
| `torch` | `src/models/lstm.py`, `src/inspect/adapters/lstm_adapter.py` | LSTM training + inference |
| `xgboost` | `src/models/xgboost_baseline.py`, `src/inspect/adapters/_xgb_worker.py` | Gradient boosting baseline |
| `plotly` | `src/data/annotate.py`, `src/inspect/viz.py` | Candlestick visualisation |
| `sklearn` | `src/inspect/stats.py`, `scripts/train_xgboost.py` | F1, confusion, train utils |
| `pytest` | `tests/` | All unit + integration tests |

## Where artifacts land

| Artifact | Path |
|----------|------|
| Trained weights | `checkpoints/<model>/<name>.{pt,ubj}` + matching `.meta.json` |
| Inspector report | `reports/inspect/<timestamp>/summary.md` + `plots/*.html` |
| Live session log | `logs/paper/<session-id>/{bars.parquet, events.jsonl, trades.sqlite}` |
| Research / plans / build logs | `.nb-suite/{research,plan,build}/<date>/<topic>.md` |
| Docs | `docs/` (this directory) |
