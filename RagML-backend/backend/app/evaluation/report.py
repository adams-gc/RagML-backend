from __future__ import annotations
from .metrics import compute_classification_metrics
from pathlib import Path
import pandas as pd

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

def generate_report(metrics: dict, filename: str = "report") -> dict:
    md = f"# Evaluation Report\n\n" \
         f"- Accuracy: {metrics['accuracy']:.3f}\n" \
         f"- Precision: {metrics['precision']:.3f}\n" \
         f"- Recall: {metrics['recall']:.3f}\n" \
         f"- F1: {metrics['f1']:.3f}\n"
    mdfile = REPORTS_DIR / f"{filename}.md"
    mdfile.write_text(md)
    (REPORTS_DIR / f"{filename}.html").write_text(f"<pre>{md}</pre>")
    df = pd.DataFrame([metrics])
    df.to_csv(REPORTS_DIR / f"{filename}.csv", index=False)
    return {"markdown": str(mdfile), "html": str(mdfile.with_suffix('.html')), "csv": str(mdfile.with_suffix('.csv'))}
