"""followup.py - rule-based next-test suggestions. Informational only."""
import datetime as dt
from typing import List, Dict, Optional

LOW_OR_HIGH = ("low", "high")


def _abn(findings, names, status=LOW_OR_HIGH):
    return any(f["parameter"] in names and f["status"] in status for f in findings)


def _indicated(preds, disease):
    p = preds.get(disease)
    return bool(p and p["prediction"])


def suggest_followups(findings: List[dict], predictions: Dict[str, Optional[dict]],
                      report_date: dt.date) -> List[dict]:
    out = []

    def add(test, why, guidance, months):
        out.append({"test": test, "reason": why, "guidance": guidance,
                    "suggested_after": (report_date + dt.timedelta(days=30 * months)).isoformat() if months else None})

    if _abn(findings, {"hba1c", "glucose_fasting", "glucose_random", "glucose_pp", "glucose_unspecified"}, ("high",)) \
            or _indicated(predictions, "Diabetes"):
        add("HbA1c", "Blood sugar above range or diabetes risk flagged",
            "Commonly repeated about every 3 months in diabetes management; confirm timing with your doctor.", 3)

    if _abn(findings, {"total_cholesterol"}, ("high",)):
        add("Lipid profile", "Cholesterol above range",
            "Often monitored every 6-12 months, depending on clinical advice.", 6)

    if _abn(findings, {"hemoglobin", "pcv", "rbc_count", "mcv", "mch", "mchc"}, ("low",)) \
            or _indicated(predictions, "Anemia"):
        add("CBC (complete blood count)", "Hemoglobin / red-cell values below range",
            "Follow-up timing depends on the cause; discuss with your doctor.", 1)

    if _abn(findings, {"serum_creatinine", "blood_urea", "bun", "potassium", "sodium"}) \
            or _indicated(predictions, "Kidney"):
        add("Kidney function test (creatinine, urea, electrolytes)", "Kidney-related values outside range",
            "Often rechecked within a few months; your doctor will advise the interval.", 3)

    if _abn(findings, {"total_bilirubin", "direct_bilirubin", "alt", "ast", "alp"}, ("high",)) \
            or _indicated(predictions, "Liver"):
        add("Liver function test", "Liver enzymes / bilirubin above range",
            "Often repeated after a few weeks to months; discuss timing with your doctor.", 2)

    if not out:
        add("Routine health check", "No out-of-range values found in this report",
            "Many adults repeat a general check-up about once a year, or as their doctor advises.", 12)
    return out