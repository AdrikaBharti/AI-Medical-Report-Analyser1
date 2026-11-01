"""history_store.py - one row per analysed report, per profile."""
import os, json, sqlite3, datetime as dt

DB = os.environ.get("HISTORY_DB", "history.db")


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            report_date TEXT NOT NULL,          -- YYYY-MM-DD (date of the test, not of upload)
            created_at TEXT NOT NULL,
            health_score REAL,
            findings TEXT NOT NULL,             -- JSON: abnormal_detector output
            predictions TEXT NOT NULL           -- JSON: {disease: {probability, prediction}}
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_profile ON reports(profile_id, report_date)")


def save_report(profile_id, report_date, findings, predictions, score):
    slim_pred = {d: {"probability": p["probability"], "prediction": p["prediction"]}
                 for d, p in predictions.items() if p}
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO reports(profile_id,report_date,created_at,health_score,findings,predictions) "
            "VALUES(?,?,?,?,?,?)",
            (profile_id, report_date, dt.datetime.utcnow().isoformat(), score,
             json.dumps(findings), json.dumps(slim_pred)))
        return cur.lastrowid


def list_reports(profile_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM reports WHERE profile_id=? ORDER BY report_date",
                         (profile_id,)).fetchall()
    return [dict(r, findings=json.loads(r["findings"]), predictions=json.loads(r["predictions"]))
            for r in rows]


def delete_profile(profile_id):          # privacy: user can wipe their data
    with _conn() as c:
        c.execute("DELETE FROM reports WHERE profile_id=?", (profile_id,))