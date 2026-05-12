
import sys, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[0]).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.xgboost_baseline import XGBoostFVGClassifier
clf = XGBoostFVGClassifier.load(sys.argv[1])
data = np.load(sys.argv[2])
vp = clf.predict_proba(data['X_val'])
tp = clf.predict_proba(data['X_test'])
np.savez(sys.argv[3], val_proba=vp, val_true=data['y_val'], test_proba=tp, test_true=data['y_test'])
print('OK')
