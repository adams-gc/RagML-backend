from dataclasses import dataclass
from typing import List, Dict
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: Dict[str, float]

def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000
    return out, dt
