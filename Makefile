.PHONY: test test-main test-xgb

# Full test suite — runs main tests + XGB baseline tests in separate processes.
# On macOS arm64, XGBoost.fit segfaults if torch was imported in the same
# session, so test_xgboost_baseline.py must run in its own pytest invocation.
test: test-main test-xgb

test-main:
	.venv/bin/pytest tests/

test-xgb:
	.venv/bin/pytest tests/models/test_xgboost_baseline.py
