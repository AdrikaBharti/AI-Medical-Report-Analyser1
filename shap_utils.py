"""
shap_utils.py - SHAP helpers shared by the training script and the API.

Kept separate on purpose: app.py only needs these two functions, and importing the whole
training script (train_final_best.py) into the server pulls in the training stack and breaks
the server whenever that script is edited or overwritten.
"""
import shap

SEED = 42   # same value as disease_model_comparison.SEED (only used to sample the SHAP background)

SHAP_KIND = {"Logistic Regression": "linear",
             "Decision Tree": "tree", "Random Forest": "tree", "XGBoost": "tree",
             "SVM": "generic", "MLP Neural Network": "generic"}


def build_explainer(algo, clf, Xt_background):
    """Right SHAP explainer for the algorithm that won. Returns (explainer, kind)."""
    kind = SHAP_KIND[algo]
    if kind == "linear":
        return shap.LinearExplainer(clf, Xt_background, feature_perturbation="interventional"), kind
    if kind == "tree":
        return shap.TreeExplainer(clf), kind
    # SVM / MLP: cost grows with background size, so keep it small (50 -> 12 is roughly 4x faster)
    bg = shap.sample(Xt_background, min(12, len(Xt_background)), random_state=SEED)
    return shap.Explainer(clf.predict_proba, bg), kind


def shap_values_positive_class(explainer, kind, Xt):
    """SHAP values for the positive class as a 2-D array (rows x features)."""
    sv = explainer(Xt, silent=True).values if kind == "generic" else explainer(Xt).values
    return sv[:, :, 1] if sv.ndim == 3 else sv