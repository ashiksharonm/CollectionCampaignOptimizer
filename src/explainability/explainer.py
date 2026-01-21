from typing import Any, Dict, List

import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

from src.features.schemas import Channel
from src.features.transformer import FeatureTransformer


class Explainer:
    """
    Train a supervised proxy model (Random Forest) on the collected interaction data
    to explain the 'Why' behind the rewards using SHAP.
    """

    def __init__(self, feature_transformer: FeatureTransformer):
        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=42
        )
        self.transformer = feature_transformer
        self.is_fitted = False
        self.explainer = None

    def fit(self, contexts: np.ndarray, actions: List[Channel], rewards: List[float]):
        """
        Fit the proxy model.
        Input: Context + Action (One-Hot)
        Target: Reward
        """
        X = self._prepare_data(contexts, actions)
        y = np.array(rewards)

        self.model.fit(X, y)
        self.is_fitted = True

        # Initialize SHAP explainer
        # We use a subset for background to speed up if data is large,
        # but here we can stick to TreeExplainer defaults or small background
        self.explainer = shap.TreeExplainer(self.model)

    def _prepare_data(self, contexts: np.ndarray, actions: List[Channel]) -> np.ndarray:
        # One-hot encode actions
        action_map = {Channel.SMS: 0, Channel.CALL: 1, Channel.FIELD_VISIT: 2}

        X_actions = np.zeros((len(actions), 3))
        for i, a in enumerate(actions):
            X_actions[i, action_map[a]] = 1.0

        return np.hstack([contexts, X_actions])

    def explain_prediction(
        self, context: np.ndarray, action: Channel
    ) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"error": "Explainer not fitted yet"}

        # Prepare single instance
        X = self._prepare_data(context.reshape(1, -1), [action])

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)

        # Helper to map feature names
        feat_names = self.transformer.get_feature_names() + [
            "ACTION_SMS",
            "ACTION_CALL",
            "ACTION_FIELD",
        ]

        # Get top contributing features
        # shap_values is typically (1, n_features)
        vals = shap_values[0]

        contributions = []
        for name, val in zip(feat_names, vals):
            contributions.append({"feature": name, "impact": float(val)})

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "predicted_reward": float(self.model.predict(X)[0]),
            "top_features": contributions[:5],
        }
