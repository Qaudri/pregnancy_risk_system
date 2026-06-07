import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

# 1. Raw dataset stats
raw = pd.read_csv("data/raw/primary_dataset.csv")
print("=== RAW DATASET ===")
print("Shape:", raw.shape)
print("Total missing:", int(raw.isnull().sum().sum()))
print("Missing per column:\n", raw.isnull().sum())
rows_with_missing = raw.isnull().any(axis=1).sum()
print("Rows containing any missing value:", int(rows_with_missing))
print("Risk Level (raw):\n", raw["Risk Level"].value_counts(dropna=False))

# 2. Cleaned dataset
clean = raw.dropna()
print("\n=== CLEANED DATASET ===")
print("Shape:", clean.shape)
print("Risk Level (cleaned):\n", clean["Risk Level"].value_counts())

# 3. Processed splits
Xtr = pd.read_csv("data/processed/X_train.csv")
ytr = pd.read_csv("data/processed/y_train.csv").values.ravel()
Xte = pd.read_csv("data/processed/X_test.csv")
yte = pd.read_csv("data/processed/y_test.csv").values.ravel()
print("\n=== SPLITS ===")
print("Train:", Xte.shape[0] + Xtr.shape[0], "->", Xtr.shape, "test:", Xte.shape)
print("Train class dist:", pd.Series(ytr).value_counts().to_dict())
print("Test class dist:", pd.Series(yte).value_counts().to_dict())
print("Features:", list(Xtr.columns))

# 4. Models + feature importances
rf  = joblib.load("models/rf_model.pkl")
xgb = joblib.load("models/xgb_model.pkl")
print("\n=== RF FEATURE IMPORTANCE ===")
print(pd.DataFrame({"f": Xtr.columns, "imp": rf.feature_importances_})
        .sort_values("imp", ascending=False).to_string(index=False))
print("\n=== XGB FEATURE IMPORTANCE ===")
print(pd.DataFrame({"f": Xtr.columns, "imp": xgb.feature_importances_})
        .sort_values("imp", ascending=False).to_string(index=False))

# 5. Test-set predictions + confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
for name, m in [("RF", rf), ("XGB", xgb)]:
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:,1]
    print(f"\n=== {name} TEST ===")
    print("Confusion matrix [[TN,FP],[FN,TP]]:\n", confusion_matrix(yte, p))
    print("ROC-AUC:", roc_auc_score(yte, pr))
    print(classification_report(yte, p, target_names=["Low","High"], digits=4))
    print(f"{name} prob — low-risk mean/std:",
          float(pr[yte==0].mean()), float(pr[yte==0].std()))
    print(f"{name} prob — high-risk mean/std:",
          float(pr[yte==1].mean()), float(pr[yte==1].std()))

# 6. Probability + importance correlations between RF and XGB
rf_pr  = rf.predict_proba(Xte)[:,1]
xgb_pr = xgb.predict_proba(Xte)[:,1]
print("\n=== RF vs XGB AGREEMENT ===")
print("Prob Pearson r:", float(np.corrcoef(rf_pr, xgb_pr)[0,1]))
rf_rank  = pd.Series(rf.feature_importances_,  index=Xtr.columns).rank()
xgb_rank = pd.Series(xgb.feature_importances_, index=Xtr.columns).rank()
print("Importance rank Pearson r:", float(rf_rank.corr(xgb_rank)))
print("Identical predictions:", int((rf.predict(Xte)==xgb.predict(Xte)).sum()), "/", len(yte))

# 7. 5-fold CV on training set (deterministic)
print("\n=== 5-FOLD CV (training set) ===")
for name, m in [("RF", rf), ("XGB", xgb)]:
    for metric in ["accuracy","precision","recall","f1"]:
        s = cross_val_score(m, Xtr, ytr, cv=5, scoring=metric, n_jobs=-1)
        print(f"{name} {metric}: {s.mean():.4f} ± {s.std():.4f}")

# 8. SHAP mean |value| (XGB)
try:
    import shap
    expl = shap.TreeExplainer(xgb)
    sv = expl.shap_values(Xte)
    if isinstance(sv, list): sv = sv[-1]
    mabs = np.abs(sv).mean(axis=0)
    print("\n=== XGB MEAN |SHAP| ===")
    print(pd.DataFrame({"f": Xte.columns, "mean_abs_shap": mabs})
            .sort_values("mean_abs_shap", ascending=False).to_string(index=False))
    # waterfall sample = highest-probability test sample
    idx = int(np.argmax(xgb.predict_proba(Xte)[:,1]))
    print("Waterfall sample index:", idx)
    print("Base value (expected_value):", float(np.array(expl.expected_value).ravel()[-1]))
    print("Sample SHAP sum:", float(sv[idx].sum()))
    print("Sample predicted prob:", float(xgb.predict_proba(Xte)[idx,1]))
except Exception as e:
    print("SHAP unavailable:", e)