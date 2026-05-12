"""window_sweep.py — Window size sensitivity analysis (Gap 7).

Usage:
  python scripts/rigor/window_sweep.py --config reports/rigor/<ts>/best_lstm_config.json
      [--windows 30 60 90 120] [--seed 42] [--output-dir reports/rigor]

Trains LSTM once per window size. Reports val + test macro F1.
Also reports effective_n per window (test_bars // W) to note confound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Window size sweep for LSTM")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--windows", nargs="+", type=int, default=[30, 60, 90, 120])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rigor"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with config_path.open() as fh:
        hyperparams = json.load(fh)
    hp = {k: v for k, v in hyperparams.items()
          if k not in ("val_macro_f1", "trial_number", "study_name", "storage")}

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    from src.data.labels import LABELLERS
    from src.data.window import SMCWindowDataset
    from src.models.lstm import FVGLSTMClassifier
    from src.training.early_stop import EarlyStop
    from src.training.loss import WeightedCE
    from src.training.train_utils import set_seed
    from src.rigor.report_utils import timestamped_dir

    data_dir = ROOT / args.data_dir
    ts_dir = timestamped_dir(ROOT / args.output_dir)

    train_df = pd.read_parquet(data_dir / "spy_h1_train.parquet")
    val_df = pd.read_parquet(data_dir / "spy_h1_val.parquet")
    test_df = pd.read_parquet(data_dir / "spy_h1_test.parquet")

    with (data_dir / "class_weights.json").open() as fh:
        cw = json.load(fh)
    weights = torch.tensor([float(cw[str(i)]) for i in range(3)], dtype=torch.float32)

    if False:  # MPS disabled: hangs on Python 3.14 with certain window configs
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    labeller = LABELLERS["fvg_valid"]()
    results = []

    for W in args.windows:
        print(f"\nWindow size W={W}...")
        set_seed(args.seed)

        train_ds = SMCWindowDataset(train_df, labeller, stride=1, window_size=W, drop_cross_session_windows=False)
        val_ds = SMCWindowDataset(val_df, labeller, stride=1, window_size=W, drop_cross_session_windows=False)
        test_ds = SMCWindowDataset(test_df, labeller, stride=1, window_size=W, drop_cross_session_windows=False)

        batch_size = int(hp.get("batch_size", 32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

        model = FVGLSTMClassifier(
            input_size=5,
            hidden_size=int(hp.get("hidden_size", 64)),
            num_layers=int(hp.get("num_layers", 2)),
            dropout=float(hp.get("dropout", 0.3)) if int(hp.get("num_layers", 2)) > 1 else 0.0,
            head_dropout=float(hp.get("head_dropout", 0.5)),
        ).to(device)

        criterion = WeightedCE(weights.to(device))
        lr = float(hp.get("lr", 1e-3))
        wd = float(hp.get("weight_decay", 1e-4))
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        early_stop = EarlyStop(patience=15, mode="max")

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(100):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = torch.as_tensor(y_batch, device=device)
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()

            model.eval()
            y_true_v, y_pred_v = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    preds = model(x).argmax(dim=-1).cpu().numpy()
                    y_pred_v.extend(preds.tolist())
                    y_true_v.extend(y.numpy().tolist() if hasattr(y, "numpy") else list(y))
            val_f1 = float(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0.0))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if early_stop.update(val_f1):
                print(f"  Early stop at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        y_true_t, y_pred_t = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=-1).cpu().numpy()
                y_pred_t.extend(preds.tolist())
                y_true_t.extend(y.numpy().tolist() if hasattr(y, "numpy") else list(y))
        test_f1 = float(f1_score(y_true_t, y_pred_t, average="macro", zero_division=0.0))

        n_test = len(test_ds)
        eff_n = n_test // W

        print(f"  W={W}: val_f1={best_val_f1:.4f}, test_f1={test_f1:.4f}, eff_n={eff_n}")
        results.append({
            "window_size": W,
            "val_macro_f1": best_val_f1,
            "test_macro_f1": test_f1,
            "n_test_windows": n_test,
            "effective_n": eff_n,
        })

    # Save results JSON
    results_path = ts_dir / "window_sweep_results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print recommendation
    best_row = max(results, key=lambda r: r["val_macro_f1"])
    print(f"\nBest window by val F1: W={best_row['window_size']} (val={best_row['val_macro_f1']:.4f})")
    print("CNN-LSTM kernel recommendation: kernel_size <= W//3 (typically 3–10)")

    # Plotly chart
    try:
        import plotly.graph_objects as go
        ws = [r["window_size"] for r in results]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ws, y=[r["val_macro_f1"] for r in results],
                                  mode="lines+markers", name="Val Macro F1"))
        fig.add_trace(go.Scatter(x=ws, y=[r["test_macro_f1"] for r in results],
                                  mode="lines+markers", name="Test Macro F1"))
        fig.update_layout(title="Window Size Sensitivity (LSTM)", xaxis_title="Window Size",
                          yaxis_title="Macro F1")
        html_path = ts_dir / "window_sweep.html"
        fig.write_html(str(html_path))
        print(f"Chart: {html_path}")
    except ImportError:
        print("plotly not available")


if __name__ == "__main__":
    main()
