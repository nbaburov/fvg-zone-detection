"""_xgb_sweep_worker.py — XGB multi-seed sweep worker (subprocess, TORCH-FREE).

IMPORTANT: Do NOT import torch or any module that imports torch.
seed_sweep.py imports torch — do NOT use it here.
This file must NOT be overwritten by multiseed_run.py at runtime.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Only import torch-free modules
import xgboost as xgb
from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-npz', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    args = parser.parse_args()

    data = np.load(args.data_npz)
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_val = np.array(data['X_val'])
    y_val = np.array(data['y_val'])
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])

    hp = json.loads(args.config)

    weights_path = ROOT / 'data' / 'processed' / 'class_weights.json'
    with weights_path.open() as fh:
        cw = json.load(fh)
    w_arr = [float(cw[str(i)]) for i in range(3)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir) / 'xgboost'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    xgb_metrics = ['mlogloss']
    rows = []

    for seed in args.seeds:
        ckpt_path = ckpt_dir / f'xgb_seed{seed}.ubj'
        meta_path = ckpt_dir / f'xgb_seed{seed}.ubj.meta.json'

        if meta_path.exists():
            with meta_path.open() as fh:
                meta = json.load(fh)
            if 'test_macro_f1' in meta and meta.get('hyperparams', {}).get('n_estimators') == hp.get('n_estimators'):
                print(f'  Seed {seed}: cached (tuned config), macro_f1={meta["test_macro_f1"]:.4f}', flush=True)
                rows.append({
                    'seed': seed,
                    'macro_f1': meta['test_macro_f1'],
                    'none_f1': meta.get('test_per_class_f1', [0, 0, 0])[0],
                    'bull_f1': meta.get('test_per_class_f1', [0, 0, 0])[1],
                    'bear_f1': meta.get('test_per_class_f1', [0, 0, 0])[2],
                })
                continue

        print(f'  Seed {seed}: training XGB (n_estimators={hp.get("n_estimators", "?")})', flush=True)
        sample_weight = np.array([w_arr[int(y)] for y in y_train], dtype=np.float32)

        clf = xgb.XGBClassifier(
            n_estimators=int(hp.get('n_estimators', 300)),
            max_depth=int(hp.get('max_depth', 4)),
            learning_rate=float(hp.get('learning_rate', 0.05)),
            min_child_weight=int(hp.get('min_child_weight', 5)),
            subsample=float(hp.get('subsample', 0.8)),
            colsample_bytree=float(hp.get('colsample_bytree', 0.8)),
            objective='multi:softprob',
            num_class=3,
            eval_metric=xgb_metrics,
            random_state=seed,
            n_jobs=1,
            tree_method='hist',
            early_stopping_rounds=30,
            verbosity=0,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight,
                eval_set=[(X_val, y_val)], verbose=False)

        y_pred = clf.predict(X_test).astype(int)
        macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0.0))
        per_class = [float(v) for v in f1_score(y_test, y_pred, average=None, zero_division=0.0)]

        clf.save_model(str(ckpt_path))
        meta_data = {
            'seed': seed, 'test_macro_f1': macro_f1, 'test_per_class_f1': per_class,
            'hyperparams': hp, 'loss_type': 'weighted_ce',
            'n_estimators_used': (clf.best_iteration + 1) if clf.best_iteration is not None else hp.get('n_estimators'),
            'optuna_study': None, 'optuna_trial_number': None, 'threshold_config': None,
        }
        with meta_path.open('w') as fh:
            json.dump(meta_data, fh, indent=2)

        np.savez(out_dir / f'xgb_seed{seed}_preds.npz', y_true=y_test, y_pred=y_pred)
        print(f'  Seed {seed}: macro_f1={macro_f1:.4f}', flush=True)
        rows.append({'seed': seed, 'macro_f1': macro_f1, 'none_f1': per_class[0],
                     'bull_f1': per_class[1], 'bear_f1': per_class[2]})

    mf = float(np.mean([r['macro_f1'] for r in rows])) if rows else 0.0
    sf = float(np.std([r['macro_f1'] for r in rows])) if rows else 0.0

    lines = [
        '# multiseed_summary_xgb_weighted_ce', '',
        '| seed | macro_f1 | none_f1 | bull_f1 | bear_f1 |',
        '|------|----------|---------|---------|---------|',
    ]
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["macro_f1"]:.4f} | {r["none_f1"]:.4f} | {r["bull_f1"]:.4f} | {r["bear_f1"]:.4f} |')
    lines.append(f'| **mean±std** | **{mf:.4f}±{sf:.4f}** | - | - | - |')
    (out_dir / 'multiseed_summary_xgb_weighted_ce.md').write_text('\n'.join(lines))
    print(f'Summary saved. Mean macro F1: {mf:.4f} +/- {sf:.4f}', flush=True)


if __name__ == '__main__':
    main()
