"""
Step 10: the predictors.

All four answer the same question: given the room's state right now and an
action, how much will the indoor temperature change over the next ten minutes?

  ZeroBaseline      always answers "no change". The score to beat; anything
                    that cannot beat this has learned nothing at all.
  TrendBaseline     assumes the last five minutes carry on for ten more. This
                    is the honest naive forecast, and it is the one that shows
                    whether the model is doing more than extrapolating.
  RidgeModel        a straight line through the nine inputs. Linear regression
                    with a penalty on large coefficients, which keeps the
                    correlated product terms from fighting each other.
  GBMModel          LightGBM: many small decision trees, each correcting the
                    last. Can bend where the physics is not a straight line.

Keeping all four matters because it answers "was the complicated one worth
it?" rather than assuming so.

Every model exposes the same predict_state(state, u, f) method, so the
controller in Step 11 can be handed any of them without changing.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, feature_vector

SLOPE_IN_IDX = FEATURE_NAMES.index("slope_in")

# Found by a grid search scored with 5-fold GroupKFold cross-validation over
# the training sessions only (never the held-out ones). The shallow trees and
# the large minimum leaf size are there because the unregularised default
# overfits badly: it reached 0.16 degC on training rows and 0.29 on held-out
# ones, which is a model memorising sessions rather than learning the room.
TUNED_GBM = dict(
    n_estimators=412,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=5.0,
)


class BaseModel:
    name = "base"

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError

    def predict_state(self, state: dict, u: float, f: float) -> float:
        """Predicted 10-minute change in T_in if we take action (u, f) from here."""
        return float(self.predict(feature_vector(state, u, f))[0])

    def as_predictor(self):
        """A plain callable, which is what ModelPolicy expects."""
        return lambda state, u, f: self.predict_state(state, u, f)


class ZeroBaseline(BaseModel):
    """Always predicts no change."""

    name = "zero"

    def predict(self, X):
        return np.zeros(len(X))


class TrendBaseline(BaseModel):
    """Assumes the last 5 minutes of indoor trend continue for 10 more.

    Note what this model cannot do: it ignores u and f entirely, so it gives
    the same answer for "shut" and "open with the fan on". It therefore has no
    opinion about which action is better, which is exactly why a controller
    needs something better than an extrapolation.
    """

    name = "trend"
    HORIZON_MIN = 10.0

    def predict(self, X):
        return np.asarray(X)[:, SLOPE_IN_IDX] * self.HORIZON_MIN


class RidgeModel(BaseModel):
    """Ridge regression on the nine inputs, with the alpha chosen by grouped CV."""

    name = "ridge"

    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.pipe = None
        self.alpha_ = None

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = make_pipeline(StandardScaler(), Ridge())
        grid = {"ridge__alpha": list(self.alphas)}
        # Cross-validate by session, for the same reason the train/test split is
        # by session: neighbouring rows are near-duplicates.
        if groups is not None and len(np.unique(groups)) >= 5:
            cv = GroupKFold(n_splits=5)
            search = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_absolute_error")
            search.fit(X, y, groups=groups)
        else:
            search = GridSearchCV(base, grid, cv=KFold(5, shuffle=True, random_state=0),
                                  scoring="neg_mean_absolute_error")
            search.fit(X, y)
        self.pipe = search.best_estimator_
        self.alpha_ = search.best_params_["ridge__alpha"]
        return self

    def predict(self, X):
        return self.pipe.predict(np.asarray(X, dtype=float))

    @property
    def coefficients(self) -> dict:
        """Coefficients in the original units of each input."""
        scaler = self.pipe.named_steps["standardscaler"]
        ridge = self.pipe.named_steps["ridge"]
        return {n: float(c / s) for n, c, s in
                zip(FEATURE_NAMES, ridge.coef_, scaler.scale_)}


class GBMModel(BaseModel):
    """LightGBM gradient-boosted trees."""

    name = "lightgbm"

    def __init__(self, **kwargs):
        self.params = dict(
            objective="l1",          # optimise the metric goal G1 is scored on
            random_state=0,
            n_jobs=-1,
            verbose=-1,
            **TUNED_GBM,
        )
        self.params.update(kwargs)
        self.model = None

    def fit(self, X, y, groups=None):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))

    @property
    def importances(self) -> dict:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        total = imp.sum() or 1.0
        return {n: float(v / total) for n, v in zip(FEATURE_NAMES, imp)}


class BlendModel(BaseModel):
    """An equal-weight average of ridge and LightGBM.

    The two make different mistakes: ridge is too stiff to bend where the
    physics is curved, LightGBM is flexible enough to chase noise. Averaging
    them cancels part of both errors. The weight is not tuned on the held-out
    sessions -- it is chosen by grouped cross-validation inside the training
    set (see scripts/step10_train.py), so the held-out score stays honest.
    """

    name = "blend"

    def __init__(self, weight_gbm: float = 0.5, ridge=None, gbm=None):
        self.weight_gbm = float(weight_gbm)
        self.ridge = ridge if ridge is not None else RidgeModel()
        self.gbm = gbm if gbm is not None else GBMModel(**TUNED_GBM)

    def fit(self, X, y, groups=None):
        self.ridge.fit(X, y, groups=groups)
        self.gbm.fit(X, y, groups=groups)
        return self

    def predict(self, X):
        w = self.weight_gbm
        return w * self.gbm.predict(X) + (1.0 - w) * self.ridge.predict(X)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


ALL_MODELS = (ZeroBaseline, TrendBaseline, RidgeModel, GBMModel, BlendModel)
