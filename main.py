
import os, warnings, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "credit_risk_dataset.csv"
TARGET = "loan_status"
RANDOM_STATE = 42
OUT_MODEL = "model.pkl"
OUT_ROC = "roc.png"

df = pd.read_csv(DATA_PATH)

cat = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
num = ["person_age", "person_income", "person_emp_length", "loan_amnt",
       "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

df["cb_person_default_on_file"] = df["cb_person_default_on_file"].astype(str).str.upper().str.strip()
df[TARGET] = df[TARGET].astype(int)

X = df[cat + num].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num),
    ],
    remainder="drop",
    n_jobs=None
)

models = {
    "logistic": LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None if hasattr(LogisticRegression, "n_jobs") else None),
    "rf": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        n_jobs=-1, random_state=RANDOM_STATE
    ),
}

def build_pipeline(clf):
    return Pipeline(steps=[
        ("prep", preprocess),
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
        ("clf", clf)
    ])

def eval_model(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    f1 = f1_score(yte, pred)
    acc = accuracy_score(yte, pred)
    return {"name": name, "pipe": pipe, "auc": auc, "f1": f1, "acc": acc, "proba": proba}

results = []
for name, clf in models.items():
    pipe = build_pipeline(clf)
    results.append(eval_model(name, pipe, X_train, y_train, X_test, y_test))

results = sorted(results, key=lambda d: d["auc"], reverse=True)
best = results[0]

print("\n=== Resultados ===")
for r in results:
    print(f"{r['name']:8s} | AUC: {r['auc']:.3f} | F1: {r['f1']:.3f} | ACC: {r['acc']:.3f}")
print(f"\nMejor: {best['name']} (AUC={best['auc']:.3f})")


fpr, tpr, _ = roc_curve(y_test, best["proba"])
plt.figure()
plt.plot(fpr, tpr, label=f"{best['name']} (AUC={best['auc']:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - Credit Risk")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=160)
print(f"ROC guardada en: {OUT_ROC}")


joblib.dump(best["pipe"], OUT_MODEL)
print(f"Modelo guardado en: {OUT_MODEL}")
