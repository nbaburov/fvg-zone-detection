#!/usr/bin/env python3
"""train_cnn_lstm.py — Thin wrapper: runs train.py with CNN-LSTM default config.

Usage:
    python scripts/training/train_cnn_lstm.py --config experiments/cnn_lstm_g1.yaml
    python scripts/training/train_cnn_lstm.py --config experiments/cnn_lstm_base.yaml --set "train.seeds=[42]" --set "train.max_epochs=2"
    python scripts/training/train_cnn_lstm.py --config experiments/cnn_lstm_base.yaml --debug

Delegates to train.py::main() with default_config pointing at cnn_lstm_g1.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.train import main

if __name__ == "__main__":
    main(default_config="experiments/cnn_lstm_g1.yaml")
