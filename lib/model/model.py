# lib/model/model.py

import joblib
import numpy as np
from typing import Dict

from lib.model.features import FEATURES

class RoundModel:
    def __init__(self, path: str):
        self.model = joblib.load(path)

    def predict_proba(self, ctx: Dict[str, float]) -> float:
        X = np.array([[ctx[name] for name in FEATURES]])

        # Probability of class "1" (UP)
        return float(self.model.predict_proba(X)[0][1])
