"""health_score.py - transparent 0-100 score. A summary of the report, NOT a clinical measure."""
from typing import List, Dict, Optional

# how much each parameter matters (default 1)
WEIGHT = {"hemoglobin": 3, "hba1c": 3, "glucose_fasting": 3, "glucose_random": 2, "glucose_pp": 2,
          "serum_creatinine": 3, "blood_urea": 2, "total_cholesterol": 2, "total_bilirubin": 2,
          "alt": 2, "ast": 2, "potassium": 2, "sodium": 2}
PENALTY = {"none": 0.0, "mild": 0.35, "moderate": 0.7, "marked": 1.0}
LAB_SHARE, MODEL_SHARE = 0.75, 0.25


def compute_health_score(findings: List[dict], predictions: Dict[str, Optional[dict]]) -> dict:
    if not findings:
        return {"score": None, "reason": "No values with a reference range were found."}

    num = den = 0.0
    drivers = []
    for f in findings:
        w = WEIGHT.get(f["parameter"], 1)
        pen = 1.0 if f["critical"] else PENALTY[f["severity"]]
        num += w * (1 - pen)
        den += w
        if pen > 0:
            drivers.append((w * pen, f["label"], f["status"]))
    lab_score = 100 * num / den

    probs = [p["probability"] for p in predictions.values() if p]
    if probs:
        model_score = 100 * (1 - max(probs))
        score = LAB_SHARE * lab_score + MODEL_SHARE * model_score
    else:
        score = lab_score

    drivers.sort(reverse=True)
    score = round(score)
    return {
        "score": score,
        "band": "Good" if score >= 80 else "Fair" if score >= 60 else "Needs attention",
        "parameters_used": len(findings),
        "lowering_factors": [f"{lbl} ({st})" for _, lbl, st in drivers[:3]],
        "note": "Summary of this report only; scores are comparable across dates only when "
                "similar tests were done. Not a medical measurement.",
    }