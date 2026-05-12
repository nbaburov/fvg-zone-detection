# MPS GPU Hangs / Python 3.14 Segfaults — Research

> **Ready for action.** No plan step needed — recommendations are concrete and actionable.
> Question: Why does nn.LSTM hang on MPS with num_layers >= 3, why does XGBoost segfault in Python 3.14 subprocesses, and what is the correct stack config for M4 Pro?
> Verdict: The MPS LSTM hang is a real unfixed PyTorch bug in the MPS graph kernel for multi-layer LSTM backprop — the only reliable fix is CPU training. Python 3.14 XGBoost segfaults stem from OpenMP + free-threaded ABI mismatch — downgrading to Python 3.12 eliminates the class of failures cleanly.

**Confidence:** High for CPU-fallback recommendation / Medium for exact Python version root cause (not lab-reproducible from search; consistent with documented pattern).
**Why this confidence:** Multiple independent PyTorch GitHub issues and forum threads confirm MPS LSTM is broken for multi-layer + dropout + backprop. XGBoost + Python 3.14 incompatibility is documented by community and inferred from free-threaded C extension ABI risk. Falsification attempted — no evidence that torch 2.11 fixes the LSTM gradient kernel assertion.
**Depth used:** Deep

---

## Project context

- LSTM trained with `nn.LSTM(num_layers=1..3)`, `dropout > 0`, using MPS device.
- Optuna 4.8.0 runs trials with `n_jobs=1` (default) — threads, not processes.
- XGBoost 3.2.0 in subprocess workers (torch-free, `n_jobs=1`) — workaround already in place.
- Python 3.14.4 Homebrew. torch 2.11.0. numpy 2.4.4.
- Sprint results: 8/50 LSTM HP trials completed before hang. Seeds 0, 123, 2024 hung in focal ablation. Window sweep killed on larger windows.

---

## Root-cause hypotheses — ranked by likelihood

### 1. MPS LSTM gradient kernel bug — `_getLSTMGradKernelDAGObject` assertion (MOST LIKELY)

**Source:** [PyTorch Forum — MPS Crash _getLSTMGradKernelDAGObject](https://discuss.pytorch.org/t/mps-crash-assertion-failed-in-getlstmgradkerneldagobject-using-lstm-on-macos-with-mps-backend/221821/1)

**Mechanism:** Apple's Metal graph kernel for LSTM backward pass asserts `shape4.size() >= 3` in `GPURNNOps.mm:2417`. This assertion fires on certain input shape + LSTM configuration combinations during `.backward()`. When combined with `torch.mps.synchronize()` (which is called implicitly at epoch boundaries), the process hangs indefinitely rather than raising — because MPS synchronize deadlocks after a kernel error ([Issue #144634](https://github.com/pytorch/pytorch/issues/144634)).

**Fit:** Yes. Confirmed on PyTorch 2.7.1 and nightly 2.9.0.dev20250801 (Aug 2025). Our num_layers=3 configs are exactly the high-layer-count configurations where shape assertions in the gradient kernel are likely to fail. The symptom (hang, not crash) matches the synchronize-after-error deadlock.

**Risk:** The bug was still open as of August 2025 with no fix merged. torch 2.11 release notes mention expanded MPS operator coverage and async error reporting — but do NOT mention a fix to the LSTM gradient kernel. No evidence this is fixed in 2.11.

---

### 2. MPS memory leak accumulating to OOM causing hang (CONTRIBUTING FACTOR)

**Source:** [Issue #145374 — MPS Memory Leak During LSTM Iterations](https://github.com/pytorch/pytorch/issues/145374)

**Mechanism:** MPS allocator does not release LSTM intermediate tensors correctly between training iterations. Memory steadily grows until MPS OOM. `torch.mps.empty_cache()` does not resolve it. Affects PyTorch 2.5.1+; no confirmed fix.

**Fit:** Partial. Explains why longer-running seeds (0, 123, 2024 requiring more epochs) hang but fast seeds (42, 17 converging quickly) complete. The window sweep hangs on larger windows (larger tensors = faster memory exhaustion).

**Risk:** Accumulates with hypothesis 1. Larger sequences or more layers = more intermediate state = faster OOM.

---

### 3. Optuna n_jobs threading + MPS GPU state corruption (SECONDARY)

**Source:** [Optuna Issue #1232 — Stop multi-thread support](https://github.com/optuna/optuna/issues/1232); [Optuna n_jobs docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)

**Mechanism:** Optuna `n_jobs > 1` uses `joblib` with `prefer="threads"`. MPS device state is not thread-safe — concurrent threads sharing the same MPS device context can corrupt command queue state. With `n_jobs=1` (our current config), trials run serially in the same thread, so this is NOT the primary cause of the observed hangs.

**Fit:** Low for our setup (n_jobs=1 already). Would be HIGH if n_jobs > 1 were used.

---

### 4. Python 3.14 free-threaded ABI / OpenMP crash in XGBoost subprocess (HIGH CONFIDENCE for XGB segfault)

**Source:** [Python 3.14 Free-Threading](https://dev.to/edgar_montano/python-314-free-threading-true-parallelism-without-the-gil-a12); [XGBoost Issue #4171 — Multiprocessing freeze](https://github.com/dmlc/xgboost/issues/4171); [XGBoost Issue #2163 — parallel hang](https://github.com/dmlc/xgboost/issues/2163)

**Mechanism:** XGBoost's C extension uses OpenMP for thread-level parallelism. Python 3.14 ships with a free-threaded build option and changes to C extension ABI. Even in the default GIL-enabled build, Homebrew Python 3.14 is a pre-release / early production Python with minimal ecosystem testing. The known XGBoost pattern is: if nthread != 1, forking after XGBoost import corrupts OpenMP thread state. Additionally, importing torch (which initialises its own thread pool) before forking is a documented segfault trigger — matching our observation.

**Fit:** High. The workarounds already in place (`n_jobs=1`, torch-free subprocess) directly match documented XGBoost OpenMP-fork mitigations. The remaining 3 skipped rigor tests still rely on n_jobs semantics that are fragile on Python 3.14.

**Risk:** Python 3.14 stable wheels for xgboost exist (3.2.0 supports 3.10–3.14), but "wheels exist" != "no ABI edge cases." The free-threaded interpreter re-enables GIL for extensions that haven't declared thread-safety — but the extension still runs in an unusual init path.

---

### 5. numpy 2.x ABI break (LOW RISK)

**Source:** [numpy/numpy Issue #26191](https://github.com/numpy/numpy/issues/26191); [PyTorch Issue #107302](https://github.com/pytorch/pytorch/issues/107302)

**Mechanism:** NumPy 2.0 broke the C ABI. Packages built against NumPy 1.x will fail at runtime with NumPy 2.x if they use the C API. PyTorch 2.2+ and XGBoost 2.x+ were rebuilt for NumPy 2.0 compatibility.

**Fit:** Low. torch 2.11 and xgboost 3.2 are both post-NumPy-2.0 releases built with 2.x in mind. Unlikely to be a live issue, but worth confirming (see validation steps).

---

## Concrete recommendations

### Fix 1: Force CPU for all LSTM training (immediate, unblocks everything)

```python
# In src/models/lstm.py or training script:
device = torch.device("cpu")  # NOT mps

# Or as env var before launching any training script:
# PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_lstm.py
# NOTE: env var must be set BEFORE Python starts — os.environ doesn't work mid-session
```

**Why:** The MPS LSTM gradient kernel bug is unresolved as of torch 2.11. CPU training on M4 Pro with batch sizes typical for this project (~117 effective windows) is fast enough — M4 Pro has high-clock CPU cores. MPS speedup doesn't apply when the training loop hangs.

**Cost:** ~2–3x slower per epoch vs theoretical MPS speedup. In practice, this project's LSTM is small (hidden_size=32, num_layers=1) and converges in <50 epochs — CPU is fine.

### Fix 2: Upgrade to Python 3.12 (medium effort, eliminates segfault class)

**Why:** Python 3.12 is the current stable LTS-equivalent for ML work as of mid-2026. All major ML wheels (torch, xgboost, numpy, shap, optuna) are battle-tested on 3.12. Python 3.14 is production-risky for C extension-heavy stacks — the community recommendation is: migrate to 3.14 late 2026 at earliest for ML projects.

**Steps:**
```bash
brew install python@3.12
python3.12 -m venv .venv-312
source .venv-312/bin/activate
pip install torch xgboost numpy pandas optuna shap alpaca-py exchange_calendars plotly scikit-learn pytest
# Verify:
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python -c "import xgboost; print(xgboost.__version__)"
```

**If downgrade is not acceptable (timeline pressure):** Keep Python 3.14 but maintain the existing subprocess isolation workaround. The 3 skipped rigor tests are not blocking for the June 20 deadline.

### Fix 3: PYTORCH_MPS_HIGH_WATERMARK_RATIO — memory pressure mitigation

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # disables hard OOM cap, lets allocator use full RAM
```

This may reduce MPS OOM-related hangs but will NOT fix the gradient kernel assertion. Use alongside CPU training fallback, not instead of it.

### Fix 4: Optuna — keep n_jobs=1, add trial timeout

```python
study.optimize(
    objective,
    n_trials=50,
    n_jobs=1,           # never increase this with MPS
    timeout=300,        # 5-min wall timeout per trial — prevents infinite hang
)
```

**Why timeout matters:** A hung MPS trial blocks Optuna forever. With a timeout, the trial is marked failed and the study continues.

### Fix 5: Window sweep — explicit CPU + reduced patience

```python
# In window sweep script, force device=cpu explicitly:
device = torch.device("cpu")
# And reduce early-stop patience to 5 (currently probably 10–15)
```

---

## Action list — exactly what to change

| Priority | Action | File | Effort |
|----------|--------|------|--------|
| P0 | Force `device = torch.device("cpu")` in LSTM training | `scripts/training/train_lstm.py`, `src/training/` | 10 min |
| P0 | Add `timeout=300` to all Optuna `study.optimize()` calls | `scripts/training/train_lstm.py` HP search | 5 min |
| P1 | Re-run focal ablation (seeds 0, 123, 2024) on CPU | — | ~30 min compute |
| P1 | Re-run window sweep on CPU with reduced patience | `scripts/training/` window sweep | ~1 hr compute |
| P2 | Create Python 3.12 venv, verify 3 skipped rigor tests pass | `.venv-312/` | 30 min |
| P3 | Add `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to `.env` or run scripts | env / Makefile | 5 min |

---

## Validation steps

**Confirm CPU fix resolves LSTM hangs:**
```python
# Minimal repro — should complete in <30s on CPU:
import torch
import torch.nn as nn

device = torch.device("cpu")
model = nn.LSTM(input_size=5, hidden_size=64, num_layers=3, dropout=0.3, batch_first=True).to(device)
x = torch.randn(32, 60, 5, device=device)
out, _ = model(x)
loss = out.mean()
loss.backward()
print("PASSED — no hang")
```

**Confirm MPS still hangs for reference (do not leave running):**
```python
device = torch.device("mps")
# same script — expect hang or assertion error during backward()
```

**Confirm XGBoost subprocess fix on Python 3.12:**
```bash
source .venv-312/bin/activate
pytest tests/rigor/ -v  # previously 3 skipped should now pass
```

**Confirm numpy ABI clean:**
```python
import numpy, torch, xgboost, shap
print(numpy.__version__, torch.__version__, xgboost.__version__, shap.__version__)
# No ImportError or ABI mismatch warnings = clean
```

---

## Falsification attempts

**Attempt 1:** "Maybe torch 2.11 fixed the LSTM MPS gradient kernel."
- Checked the 2.11 release blog. MPS improvements listed: async error reporting, new distribution ops, grid_sampler_2d, baddbmm extensions. No mention of LSTM gradient kernel fix.
- Forum thread from Aug 2025 confirms nightly 2.9.0 still broken. No evidence 2.11 resolved it.
- Verdict: Recommendation to use CPU survives.

**Attempt 2:** "Maybe Python 3.14 XGBoost segfaults are isolated to n_jobs=-1 and are already fully fixed by n_jobs=1 workaround."
- The existing workaround (n_jobs=1 + torch-free subprocess) does resolve the production path.
- The 3 skipped tests suggest edge cases remain. But they are non-blocking for the deadline.
- Verdict: Python 3.12 upgrade is a P2, not P0.

**Attempt 3:** "Maybe PYTORCH_ENABLE_MPS_FALLBACK=1 would make LSTM run on CPU automatically for broken ops."
- The fallback only triggers for ops that raise `NotImplementedError` (unsupported ops). The LSTM gradient kernel assertion is a Metal crash, not a NotImplementedError — fallback does NOT trigger.
- Verdict: PYTORCH_ENABLE_MPS_FALLBACK=1 does not fix this hang.

---

## What I couldn't verify

- Whether the specific `_getLSTMGradKernelDAGObject` assertion fires for unidirectional (non-bidirectional) LSTM with num_layers=3 specifically, vs only bidirectional. The forum thread describes a bidirectional model; the project uses unidirectional. The hang is observed empirically (build log) but the exact kernel assertion path for unidirectional multi-layer is not confirmed from sources.
- Whether torch 2.11 introduced any silent LSTM MPS regression vs 2.10 (only release blog was checked, not full diff).
- NumPy 2.4.4 is very recent (current version as of May 2026). Whether SHAP 0.51.0 is built against NumPy 2.x is not confirmed — worth a quick `python -c "import shap"` check for import errors.

---

## Sources

- [PyTorch Forum: MPS LSTM _getLSTMGradKernelDAGObject crash](https://discuss.pytorch.org/t/mps-crash-assertion-failed-in-getlstmgradkerneldagobject-using-lstm-on-macos-with-mps-backend/221821/1)
- [PyTorch Issue #144634: torch.mps.synchronize hangs on error](https://github.com/pytorch/pytorch/issues/144634)
- [PyTorch Issue #145374: MPS Memory Leak in LSTM](https://github.com/pytorch/pytorch/issues/145374)
- [PyTorch Issue #173640: LSTM dropout output collapse on MPS](https://github.com/pytorch/pytorch/issues/173640)
- [PyTorch Issue #90421: LSTM for MPS backend is broken](https://github.com/pytorch/pytorch/issues/90421)
- [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch MPS Environment Variables — 2.11 docs](https://docs.pytorch.org/docs/2.11/mps_environment_variables.html)
- [XGBoost Issue #4171: Multiprocessing freeze](https://github.com/dmlc/xgboost/issues/4171)
- [XGBoost Issue #2163: parallel hang nthread!=1](https://github.com/dmlc/xgboost/issues/2163)
- [Optuna Issue #1232: Multi-thread support concerns](https://github.com/optuna/optuna/issues/1232)
- [numpy/numpy Issue #26191: Ecosystem compatibility with numpy 2.0](https://github.com/numpy/numpy/issues/26191)
- [invisiblefriends.net: Using Python on Apple Silicon Macs in 2026](https://www.invisiblefriends.net/using-python-on-apple-silicon-macs-in-2026/)
- [Python 3.14 Free-Threading — DEV Community](https://dev.to/edgar_montano/python-314-free-threading-true-parallelism-without-the-gil-a12)
