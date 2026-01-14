# lib/model/model.py

import joblib
import numpy as np
from typing import Dict

class RoundModel:
    def __init__(self, path: str):
        self.model = joblib.load(path)

    def predict_proba(self, ctx: Dict[str, float]) -> float:
        X = np.array([[
            ctx["ret_1m"],
            ctx["ret_5m"],
            ctx["ret_15m"],
            ctx["vol_15m"],
            ctx["vol_60m"],
            ctx["range_15m"],
            ctx["volume_15m"],
        ]])

        # Probability of class "1" (UP)
        return float(self.model.predict_proba(X)[0][1])
