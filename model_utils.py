"""
model_utils.py — Shared model classes

All classifier wrappers must live here so joblib can unpickle them from any
entry point (train_classifier.py, evaluate_system.py, grid_monitor.py, etc.).
"""

import numpy as np


class IsotonicCalibratedClassifier:
    """
    Wraps a pre-fitted LightGBM classifier with isotonic probability calibration.

    sklearn >= 1.2 removed cv='prefit' from CalibratedClassifierCV, so calibration
    is implemented directly:
      1. Base LightGBM generates raw P(Attack) on a held-out calibration set.
      2. IsotonicRegression maps those raw scores to well-calibrated probabilities.
      3. predict_proba() applies both steps at inference time.

    Fully picklable via joblib -- drop-in replacement for the raw LightGBM model.
    Attack=0, Natural=1 (LabelEncoder alphabetical order).
    """
    def __init__(self, base_clf, isotonic_regressor):
        self.base_clf = base_clf
        self.isotonic = isotonic_regressor

    def predict_proba(self, X):
        raw        = self.base_clf.predict_proba(X)[:, 0]   # P(Attack), uncalibrated
        calibrated = self.isotonic.predict(raw)               # corrected probabilities
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([calibrated, 1.0 - calibrated])

    def predict(self, X):
        # Attack=0, Natural=1 -- return 0 when P(Attack) >= 0.5, else 1
        return np.where(self.predict_proba(X)[:, 0] >= 0.5, 0, 1)


class EnsembleCalibratedClassifier:
    """
    LightGBM + XGBoost stacked ensemble with SHAP-guided feature selection.

    Pipeline:
      1. Both base models are trained on a SHAP-selected feature subset.
      2. A LogisticRegression meta-learner combines their P(Attack) outputs.
      3. predict_proba() applies feature selection internally so callers always
         pass the full-width feature array — no changes needed at call sites.

    SHAP compatibility:
      - base_clf  : the LightGBM model (used by shap.TreeExplainer).
      - get_shap_input(X) : returns the feature-selected slice of X.
      - map_shap_to_all(shap_1d, n) : zero-pads SHAP values back to full width.

    Attack=0, Natural=1 throughout (LabelEncoder alphabetical order).
    """

    def __init__(self, lgb_clf, xgb_clf, meta_learner,
                 selected_feature_indices, selected_feature_names,
                 optimal_threshold=0.5):
        self.base_clf = lgb_clf                                         # kept for SHAP
        self.xgb_clf  = xgb_clf
        self.meta_learner = meta_learner
        self.selected_feature_indices = np.array(selected_feature_indices, dtype=int)
        self.selected_feature_names   = list(selected_feature_names)
        self.optimal_threshold        = float(optimal_threshold)

    def _select(self, X):
        if hasattr(X, 'iloc'):   # pandas DataFrame
            return X.iloc[:, self.selected_feature_indices].values
        return X[:, self.selected_feature_indices]

    def predict_proba(self, X):
        X_sel   = self._select(X)
        lgb_p   = self.base_clf.predict_proba(X_sel)[:, 0]   # P(Attack)
        xgb_p   = self.xgb_clf.predict_proba(X_sel)[:, 0]   # P(Attack)
        stacked = np.column_stack([lgb_p, xgb_p])
        # meta_learner classes: 0=Natural, 1=Attack → col 1 is P(Attack)
        p_atk   = self.meta_learner.predict_proba(stacked)[:, 1]
        p_atk   = np.clip(p_atk, 0.0, 1.0)
        return np.column_stack([p_atk, 1.0 - p_atk])

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 0] >= self.optimal_threshold, 0, 1)

    def get_shap_input(self, X):
        """Return feature-selected slice of X (numpy array) for shap.TreeExplainer(base_clf)."""
        result = self._select(X)
        if hasattr(result, 'values'):
            return result.values
        return result

    def map_shap_to_all(self, shap_1d, n_all_features):
        """Zero-pad selected-feature SHAP values back to the full feature width."""
        full = np.zeros(n_all_features)
        full[self.selected_feature_indices] = shap_1d
        return full


class CascadedScenarioClassifier:
    """
    Cascade of N per-scenario OvR binary LightGBM classifiers.

    Each classifier is trained on one experimental run (CSV file) from the
    MSU/ORNL dataset.  For classifier i, the positive class is Attack samples
    from file i; the negative class is everything else (other attacks + Natural).

    At inference time classifiers are evaluated in `scenario_order` (sorted by
    validation F1, highest first).  The first classifier to exceed its per-
    scenario threshold wins and its 1-based scenario index is returned.
    If no classifier fires, the sample is classified as Natural (label 0).

    scenario_names maps 1-based scenario index → human-readable label.
    """

    def __init__(self, classifiers, thresholds, scenario_order,
                 selected_feature_indices, n_scenarios, scenario_names=None):
        self.classifiers              = classifiers
        self.thresholds               = thresholds
        self.scenario_order           = scenario_order          # list of 0-based indices
        self.selected_feature_indices = np.array(selected_feature_indices, dtype=int)
        self.n_scenarios              = n_scenarios
        self.scenario_names           = scenario_names or {i+1: f"Scenario_{i+1}"
                                                           for i in range(n_scenarios)}

    def _select(self, X):
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.selected_feature_indices].values
        return X[:, self.selected_feature_indices]

    def predict(self, X):
        """
        Returns int array: 0 = Natural, 1..N = scenario index.
        Cascade stops at the first scenario that exceeds its threshold.
        """
        X_sel      = self._select(X)
        n          = X_sel.shape[0]
        preds      = np.zeros(n, dtype=int)
        unresolved = np.ones(n, dtype=bool)

        for idx in self.scenario_order:
            if not unresolved.any():
                break
            proba      = self.classifiers[idx].predict_proba(X_sel[unresolved])[:, 1]
            fired      = proba >= self.thresholds[idx]
            positions  = np.where(unresolved)[0][fired]
            preds[positions]      = idx + 1   # 1-based scenario index
            unresolved[positions] = False

        return preds

    def predict_proba_all(self, X):
        """
        Returns (n_samples, n_scenarios+1) array.
        Column 0 = P(Natural) = 1 - max(scenario probs).
        Columns 1..N = per-scenario probabilities.
        """
        X_sel  = self._select(X)
        probas = np.zeros((X_sel.shape[0], self.n_scenarios + 1))
        for idx in range(self.n_scenarios):
            probas[:, idx + 1] = self.classifiers[idx].predict_proba(X_sel)[:, 1]
        probas[:, 0] = np.clip(1.0 - probas[:, 1:].max(axis=1), 0.0, 1.0)
        return probas

    def scenario_label(self, scenario_idx):
        """Human-readable name for a 1-based scenario index (0 → 'Natural')."""
        return "Natural" if scenario_idx == 0 else self.scenario_names.get(scenario_idx, f"Scenario_{scenario_idx}")
