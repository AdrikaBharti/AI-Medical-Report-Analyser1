"""
External validation of the 5 trained disease models
====================================================
Takes the .joblib artifacts produced by train_final.py / train_final_best.py,
ALIGNS each new dataset to the exact schema the models were trained on, scores
it with the SAVED models (no refitting, no new scaler/imputer), and compares the
result with the internal (out-of-fold) numbers stored in each artifact.

Folder layout expected (override with env vars or CLI flags):
    external_datasets/   anemia.csv, diabetes_prediction_dataset.csv, heart.csv,
                         kidney_disease_dataset.csv, Liver_Patient_Dataset (LPD)_train.csv
    outputs/             final_models_best/*.joblib  (or final_models/)
                         cleaned_data/*_clean.csv    (written by disease_model_comparison.py)

Run:
    python external_validation.py
    python external_validation.py --models-dir outputs/final_models --ext-dir my_data

Outputs  (outputs/external_validation/):
    comparison_table.csv        internal vs external metrics + verdict
    feature_shift_<disease>.csv training-vs-external feature distributions
    comparison_bars.png         grouped bar chart
    external_roc.png            ROC curves on the external sets
"""
import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
                             precision_score, recall_score, f1_score, accuracy_score)

warnings.filterwarnings("ignore")
OUT = os.environ.get("OUTDIR", "outputs")
EXT = os.environ.get("EXTERNAL", "external_datasets")
SEED = 42


# =============================================================================
# 1. SMALL HELPERS
# =============================================================================
def read_csv_any(path):
    """Read a CSV whatever its encoding, and clean the column names
    (non-breaking spaces, double spaces, leading/trailing blanks)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = (df.columns.astype(str).str.replace("\xa0", " ", regex=False)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return df


def find_file(pattern, ext_dir):
    hits = sorted(glob.glob(os.path.join(ext_dir, pattern)))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' in '{ext_dir}'")
    return hits[0]


def finish(X, y):
    """Drop exact duplicate rows (training did the same) and reset the index."""
    X = X.copy()
    X["__y"] = pd.Series(y).values
    X = X.drop_duplicates().reset_index(drop=True)
    y = X.pop("__y").astype(int)
    return X, y


# =============================================================================
# 2. FEATURE ALIGNMENT  - one function per disease
#    Each returns (X, y, notes):  X uses the TRAINING column names, units and
#    encodings. Features that don't exist in the new dataset are simply left out;
#    they are added back as NaN later, and the saved imputer fills them with the
#    TRAINING median (exactly what the app does for an optional field left blank).
# =============================================================================
def align_diabetes(df):
    """Training (Pima): Glucose, Insulin, Age, BMI, BloodPressure, Pregnancies.
    New dataset only has glucose, age, BMI -> Insulin / BloodPressure / Pregnancies = NaN."""
    df = df.drop_duplicates()
    df = df[df["age"] >= 21]                       # Pima = adults 21+; new set includes children
    X = pd.DataFrame({
        "Glucose": df["blood_glucose_level"].astype(float),   # mg/dL in both (see notes: NOT the same test)
        "Age":     df["age"].astype(float),
        "BMI":     df["bmi"].astype(float),
    })
    for c in ["Glucose", "BMI"]:                   # training rule: 0 is impossible -> missing
        X.loc[X[c] <= 0, c] = np.nan
    X, y = finish(X, df["diabetes"])
    return X, y, ("Only Glucose/Age/BMI exist in the new set (Insulin, BloodPressure, Pregnancies "
                  "imputed). Pima glucose = 2-h OGTT; new set = random blood glucose. Children removed.")


def align_heart(df):
    """Training (UCI): age, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak.
    cp / restecg are one-hot encoded from STRINGS in training -> spellings must match exactly."""
    df = df.drop_duplicates()
    cp_map  = {"TA": "typical angina", "ATA": "atypical angina",
               "NAP": "non-anginal",   "ASY": "asymptomatic"}
    ecg_map = {"Normal": "normal", "ST": "st-t abnormality", "LVH": "lv hypertrophy"}
    X = pd.DataFrame({
        "age":      df["Age"].astype(float),
        "cp":       df["ChestPainType"].map(cp_map),
        "trestbps": df["RestingBP"].astype(float).replace(0, np.nan),     # 0 = missing (as in training)
        "chol":     df["Cholesterol"].astype(float).replace(0, np.nan),   # 0 = missing (172 rows!)
        "fbs":      df["FastingBS"].astype(float),
        "restecg":  df["RestingECG"].map(ecg_map),
        "thalch":   df["MaxHR"].astype(float),                            # renamed
        "exang":    df["ExerciseAngina"].map({"Y": 1.0, "N": 0.0}),
        "oldpeak":  df["Oldpeak"].astype(float),
    })
    X, y = finish(X, df["HeartDisease"])
    return X, y, "All 9 features available. cp/restecg strings mapped to the UCI spellings."


def align_kidney(df):
    """Training (UCI CKD): 17 features. All exist in the new set under long names.
    Binary text columns were encoded 0/1 in training (normal/notpresent=0)."""
    df = df.drop_duplicates()
    abn = {"normal": 0.0, "abnormal": 1.0}
    pres = {"not present": 0.0, "present": 1.0}      # note the SPACE in the new file ('notpresent' in UCI)
    num = {"age": "Age of the patient", "sg": "Specific gravity of urine",
           "al": "Albumin in urine", "su": "Sugar in urine",
           "bgr": "Random blood glucose level (mg/dl)", "bu": "Blood urea (mg/dl)",
           "sc": "Serum creatinine (mg/dl)", "sod": "Sodium level (mEq/L)",
           "pot": "Potassium level (mEq/L)", "hemo": "Hemoglobin level (gms)",
           "pcv": "Packed cell volume (%)", "wbcc": "White blood cell count (cells/cumm)",
           "rbcc": "Red blood cell count (millions/cumm)"}
    X = pd.DataFrame({k: df[v].astype(float) for k, v in num.items()})
    X["rbc"] = df["Red blood cells in urine"].str.strip().str.lower().map(abn)
    X["pc"]  = df["Pus cells in urine"].str.strip().str.lower().map(abn)
    X["pcc"] = df["Pus cell clumps in urine"].str.strip().str.lower().map(pres)
    X["ba"]  = df["Bacteria in urine"].str.strip().str.lower().map(pres)
    y = (df["Target"] != "No_Disease").astype(int)    # No_Disease=0; Low/Moderate/High/Severe = 1
    X, y = finish(X, y)
    return X, y, "All 17 features available (units identical). 5-level Target binarised: No_Disease=0, rest=1."


def align_liver(df):
    """Training (ILPD): Age, Gender(M=1,F=0), Total/Direct Bilirubin, Alkaline_Phosphotase,
    ALT_SGPT, AST_SGOT, Total_Proteins, Albumin, AG_Ratio.  Label: 1 = liver patient, 2 = not."""
    df = df.drop_duplicates()
    X = pd.DataFrame({
        "Age":                  df["Age of the patient"].astype(float),
        "Gender":               df["Gender of the patient"].map({"Male": 1.0, "Female": 0.0}),
        "Total_Bilirubin":      df["Total Bilirubin"].astype(float),
        "Direct_Bilirubin":     df["Direct Bilirubin"].astype(float),
        "Alkaline_Phosphotase": df["Alkphos Alkaline Phosphotase"].astype(float),
        "ALT_SGPT":             df["Sgpt Alamine Aminotransferase"].astype(float),
        "AST_SGOT":             df["Sgot Aspartate Aminotransferase"].astype(float),
        "Total_Proteins":       df["Total Protiens"].astype(float),          # (sic) typo in source file
        "Albumin":              df["ALB Albumin"].astype(float),
        "AG_Ratio":             df["A/G Ratio Albumin and Globulin Ratio"].astype(float),
    })
    y = (df["Result"] == 1).astype(int)
    X, y = finish(X, y)
    return X, y, "All 10 features available. Result 1/2 -> 1/0 exactly as in training."


def align_anemia(df):
    """Training: Gender(M=1,F=0), Age, Hemoglobin, RBC, PCV_HCT, MCV, MCH, MCHC.
    New dataset has only Gender, Hemoglobin, MCV, MCH, MCHC.
    !! Its Gender is coded 0=male, 1=female (opposite of training) -> flipped below.
       Evidence: anemia rate is 57% for code 1 vs 35% for code 0.  Set FLIP_ANEMIA_GENDER=False
       if you know your copy of the file is coded the other way."""
    df = df.drop_duplicates()
    gender = 1.0 - df["Gender"].astype(float) if FLIP_ANEMIA_GENDER else df["Gender"].astype(float)
    X = pd.DataFrame({"Gender": gender,
                      "Hemoglobin": df["Hemoglobin"].astype(float),
                      "MCV": df["MCV"].astype(float),
                      "MCH": df["MCH"].astype(float),
                      "MCHC": df["MCHC"].astype(float)})
    X, y = finish(X, df["Result"])
    return X, y, "Age / RBC / PCV_HCT missing (imputed). Gender recoded to M=1,F=0. Duplicates removed."


FLIP_ANEMIA_GENDER = True

DATASETS = {   # disease -> (aligner, filename pattern inside EXT)
    "Anemia":   (align_anemia,   "*anemia*.csv"),
    "Diabetes": (align_diabetes, "*diabetes*.csv"),
    "Heart":    (align_heart,    "*heart*.csv"),
    "Kidney":   (align_kidney,   "*kidney*.csv"),
    "Liver":    (align_liver,    "*Liver*.csv"),
}


# =============================================================================
# 3. SAFETY CHECKS ON THE ALIGNMENT
# =============================================================================
def check_categories(art, X):
    """OneHotEncoder(handle_unknown='ignore') silently turns an unseen spelling into all-zeros.
    Compare every categorical value against the categories the encoder learned in training."""
    pre = art["pipeline"].named_steps["pre"]
    if "cat" not in pre.named_transformers_:
        return
    known = pre.named_transformers_["cat"].named_steps["oh"].categories_
    for col, cats in zip(art["categorical_features"], known):
        bad = set(X[col].dropna().unique()) - set(cats)
        if bad:
            raise ValueError(f"[{art['disease']}] column '{col}': values {bad} were never seen in "
                             f"training (training categories: {list(cats)}). Fix the mapping.")


def uni_auc(X, y, cols):
    """Univariate AUC of each numeric column vs the label (0.5 = no relationship)."""
    out = {}
    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce")
        m = s.notna()
        if m.sum() >= 30 and s[m].nunique() > 1 and y[m].nunique() == 2:
            out[c] = roc_auc_score(y[m], s[m])
    return out


def load_training_frame(name):
    p = os.path.join(OUT, "cleaned_data", f"{name.lower()}_clean.csv")
    if not os.path.exists(p):
        return None, None
    t = pd.read_csv(p)
    return t.drop(columns="target"), t["target"].astype(int)


def overlap_mask(Xtr, X):
    """Approximate 'is this external row really a training row?' check.
    Compares only the continuous measurement columns (>= 10 distinct values, NOT age/sex, because
    re-published copies often alter demographics) and ignores cells that are blank in either row.
    A row counts as overlapping if it matches some training row on ALL compared columns and at
    least 4 columns were comparable.  (Liver 'LPD' is ILPD re-published with altered age/sex.)"""
    demo = {"age", "gender", "sex"}
    cols = [c for c in X.columns if c in Xtr.columns and c.lower() not in demo
            and pd.api.types.is_numeric_dtype(Xtr[c]) and Xtr[c].nunique() >= 10 and X[c].notna().any()]
    if len(cols) < 4:
        return None                                  # too few shared measurements to judge
    A = Xtr[cols].astype(float).round(2).values
    B = X[cols].astype(float).round(2).values
    hit = np.zeros(len(B), bool)
    for s in range(0, len(B), 500):
        b, a = B[s:s + 500, None, :], A[None, :, :]
        both = ~np.isnan(b) & ~np.isnan(a)
        compared, matches = both.sum(2), ((b == a) & both).sum(2)
        hit[s:s + 500] = ((compared >= 4) & (matches == compared)).any(axis=1)
    return hit


def diagnostics(name, X, y, Xtr, ytr, out_dir):
    """Everything that helps explain WHY performance changed (or why it can't be trusted)."""
    d = dict(overlap=None, no_signal=False, flips=[], shifts=[])
    ext_auc = uni_auc(X, y, X.columns)
    if ext_auc:
        d["max_dev"] = max(abs(a - 0.5) for a in ext_auc.values())
        d["no_signal"] = d["max_dev"] < 0.06       # no feature separates the classes at all
    if Xtr is None:
        return d
    d["overlap"] = overlap_mask(Xtr, X)
    tr_auc = uni_auc(Xtr, ytr, X.columns)
    d["flips"] = [c for c in ext_auc if c in tr_auc and (tr_auc[c] - .5) * (ext_auc[c] - .5) < 0
                  and min(abs(tr_auc[c] - .5), abs(ext_auc[c] - .5)) > 0.10]
    rows = []
    for c in X.columns:
        if c in Xtr.columns and pd.api.types.is_numeric_dtype(Xtr[c]) and X[c].notna().any():
            sd = Xtr[c].std()
            rows.append(dict(feature=c, train_mean=Xtr[c].mean(), external_mean=X[c].mean(),
                             train_sd=sd, std_mean_diff=(X[c].mean() - Xtr[c].mean()) / sd if sd else np.nan,
                             train_missing_pct=100 * Xtr[c].isna().mean(),
                             external_missing_pct=100 * X[c].isna().mean()))
    shift = pd.DataFrame(rows)
    shift.to_csv(os.path.join(out_dir, f"feature_shift_{name.lower()}.csv"), index=False)
    d["shifts"] = shift.loc[shift.std_mean_diff.abs() > 0.5, "feature"].tolist() if len(shift) else []
    return d


# =============================================================================
# 4. SCORING WITH THE SAVED MODEL  (transform only - nothing is refitted)
# =============================================================================
def auc_ci(y, s, n=400):
    rng, y, s, v = np.random.default_rng(SEED), np.asarray(y), np.asarray(s), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def score(art, X, y):
    X = X[art["feature_names"]]
    raw = art["pipeline"].predict_proba(X)[:, 1]        # saved imputer + scaler/encoder + classifier
    cal = art["calibrator"].predict_proba(X)[:, 1]      # what the app shows to the user
    pred = (cal >= art["threshold"]).astype(int)        # same rule the app applies
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    lo, hi = auc_ci(y, raw)
    return dict(N=len(y), Prevalence=y.mean(), AUC=roc_auc_score(y, raw), AUC_CI=f"{lo:.3f}-{hi:.3f}",
                Recall=rec, Specificity=spec, Precision=precision_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0), Accuracy=accuracy_score(y, pred),
                Balanced_Acc=(rec + spec) / 2, Mean_pred_risk=cal.mean(),
                Brier=brier_score_loss(y, cal), _raw=raw)


def internal_metrics(art):
    """Out-of-fold numbers stored by train_final*.py (threshold chosen on the same OOF data,
    so these are slightly optimistic - say so in the report)."""
    tm = art["threshold_metrics"]
    tn, fp, fn, tp = np.array(tm["confusion_matrix"]).ravel()
    rec, spec = tm["recall"], tn / (tn + fp)
    return dict(N=art.get("trained_on_n_rows", tn + fp + fn + tp), Prevalence=(fn + tp) / (tn + fp + fn + tp),
                AUC=art["oof_roc_auc"], Recall=rec, Specificity=spec, Precision=tm["precision"],
                F1=tm["f1"], Accuracy=tm["accuracy"], Balanced_Acc=(rec + spec) / 2)


def verdict(d_auc, overlap_pct, no_signal, subset):
    if no_signal:
        return "UNINFORMATIVE - no input feature is related to the external labels"
    if subset == "all rows" and overlap_pct >= 20:
        return f"NOT INDEPENDENT - {overlap_pct:.0f}% of rows are also in the training data"
    if d_auc >= -0.03:
        return "Generalises well (AUC drop < 0.03)"
    if d_auc >= -0.10:
        return "Moderate drop (AUC 0.03-0.10 lower)"
    return "Large drop - poor transfer (AUC > 0.10 lower)"


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    default_models = os.path.join(OUT, "final_models_best")
    if not os.path.isdir(default_models):
        default_models = os.path.join(OUT, "final_models")
    ap.add_argument("--models-dir", default=default_models)
    ap.add_argument("--ext-dir", default=EXT)
    ap.add_argument("--out-dir", default=os.path.join(OUT, "external_validation"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows, roc_data = [], {}
    for name, (aligner, pattern) in DATASETS.items():
        art = joblib.load(os.path.join(a.models_dir, f"{name.lower()}_final.joblib"))
        df = read_csv_any(find_file(pattern, a.ext_dir))

        X, y, notes = aligner(df)
        X = X.reindex(columns=art["feature_names"])             # exact training columns + order
        missing = [c for c in X.columns if X[c].isna().all()]   # absent in the new dataset
        check_categories(art, X)
        n_req = int(X[art["required_fields"]].isna().any(axis=1).sum())

        Xtr, ytr = load_training_frame(name)
        diag = diagnostics(name, X, y, Xtr, ytr, a.out_dir)
        ov = diag["overlap"]
        ov_pct = 100 * ov.mean() if ov is not None else 0.0

        print(f"\n[{name}] {notes}")
        print(f"   rows={len(X)}  prevalence={y.mean():.1%}  features missing entirely: {missing or 'none'}"
              f"  | rows lacking a required field: {n_req}")
        if diag["flips"]:
            print(f"   !! feature/label direction FLIPPED vs training for {diag['flips']} -> check encoding")
        if diag["shifts"]:
            print(f"   large distribution shift (|std diff|>0.5): {diag['shifts']}")
        if ov is not None:
            print(f"   overlap with training rows: {ov.sum()} ({ov_pct:.1f}%)")

        subsets = [("all rows", np.ones(len(X), bool))]
        if ov is not None and 0 < ov.sum() < len(X):
            subsets.append(("overlap removed", ~ov))
        base = internal_metrics(art)
        for label, m in subsets:
            Xs, ys = X[m].reset_index(drop=True), y[m].reset_index(drop=True)
            if ys.nunique() < 2 or len(ys) < 30:
                continue
            ext = score(art, Xs, ys)
            if label == "all rows":
                roc_data[name] = (ys, ext["_raw"], ext["AUC"])
            r = dict(Disease=name, Algorithm=art["algorithm"], Subset=label,
                     N_internal=base["N"], N_external=ext["N"],
                     Prevalence_internal=base["Prevalence"], Prevalence_external=ext["Prevalence"],
                     Features_missing_in_external=", ".join(missing) or "-",
                     Overlap_pct=round(ov_pct, 1) if label == "all rows" else 0.0)
            for k in ["AUC", "Recall", "Specificity", "Precision", "F1", "Accuracy", "Balanced_Acc"]:
                r[f"{k}_internal"], r[f"{k}_external"] = base[k], ext[k]
            r.update(AUC_external_95CI=ext["AUC_CI"], Delta_AUC=ext["AUC"] - base["AUC"],
                     Delta_Recall=ext["Recall"] - base["Recall"],
                     Mean_pred_risk_external=ext["Mean_pred_risk"], Brier_external=ext["Brier"],
                     Verdict=verdict(ext["AUC"] - base["AUC"], ov_pct, diag["no_signal"], label))
            rows.append(r)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(os.path.join(a.out_dir, "comparison_table.csv"), index=False)

    show = table[["Disease", "Subset", "N_external", "AUC_internal", "AUC_external", "Recall_internal",
                  "Recall_external", "Specificity_internal", "Specificity_external", "Verdict"]]
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print("\n=== INTERNAL (out-of-fold) vs EXTERNAL ===")
    print(show.round(3).to_string(index=False))

    plot_bars(table[table.Subset == "all rows"], a.out_dir)
    plot_roc(roc_data, a.out_dir)
    print(f"\nSaved to {a.out_dir}/")


# =============================================================================
# 6. PLOTS
# =============================================================================
def plot_bars(t, out_dir):
    metrics = ["AUC", "Recall", "Specificity", "F1"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    x, w = np.arange(len(t)), 0.38
    for ax, m in zip(axes, metrics):
        b1 = ax.bar(x - w / 2, t[f"{m}_internal"], w, label="Internal (5-fold OOF)", color="#1F4E78")
        b2 = ax.bar(x + w / 2, t[f"{m}_external"], w, label="External dataset", color="#E07B39")
        ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(t.Disease); ax.set_title(m, fontweight="bold")
        ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Internal vs external validation  (prevalence differs between sets - read F1/precision with care)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "comparison_bars.png"), dpi=150); plt.close(fig)


def plot_roc(roc_data, out_dir):
    fig, axes = plt.subplots(1, len(roc_data), figsize=(4 * len(roc_data), 4))
    for ax, (name, (y, s, auc)) in zip(np.atleast_1d(axes), roc_data.items()):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color="#E07B39", lw=2); ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{name}  (AUC {auc:.3f})", fontweight="bold")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "external_roc.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()