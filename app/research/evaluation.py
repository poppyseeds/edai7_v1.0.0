"""Modality-specific diagnostics and empirical distance-attack controls."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error


def sequence_metrics(real: np.ndarray, synthetic: np.ndarray) -> dict:
    if real.ndim != 3 or synthetic.ndim != 3 or real.shape[1:] != synthetic.shape[1:]:
        raise ValueError("Temporal metrics require matching [sequence, timestep, feature] shapes.")
    if not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("Temporal metrics require finite measurements.")
    per_feature = []
    for feature in range(real.shape[2]):
        a, b = real[:, :, feature], synthetic[:, :, feature]
        scale = max(float(a.std()), 1e-8)
        acf_errors = []
        for lag in range(1, min(10, real.shape[1]//4)+1):
            def acf(values):
                centered = values-values.mean(1, keepdims=True)
                denominator = (centered**2).sum(1)
                valid = denominator > 1e-12
                return ((centered[:, :-lag]*centered[:, lag:]).sum(1)[valid] / denominator[valid]).mean() if valid.any() else None
            ra, rb = acf(a), acf(b)
            if ra is not None and rb is not None:
                acf_errors.append(abs(float(ra-rb)))
        time = np.arange(a.shape[1], dtype=float)
        time -= time.mean()
        slope_a, slope_b = a @ time / (time @ time), b @ time / (time @ time)
        power_a = (np.abs(np.fft.rfft(a-a.mean(1, keepdims=True), axis=1))**2).mean(0)
        power_b = (np.abs(np.fft.rfft(b-b.mean(1, keepdims=True), axis=1))**2).mean(0)
        spectral = float(jensenshannon(power_a/power_a.sum(), power_b/power_b.sum(), base=2)**2) if power_a.sum() > 0 and power_b.sum() > 0 else None
        per_feature.append({"feature_index": feature, "ks_statistic": float(ks_2samp(a.ravel(), b.ravel()).statistic),
            "normalized_mean_error": abs(float(a.mean()-b.mean()))/scale,
            "normalized_variance_error": abs(float(a.var()-b.var()))/(float(a.var())+1e-8),
            "acf_error": float(np.mean(acf_errors)) if acf_errors else None,
            "normalized_trend_error": abs(float(slope_a.mean()-slope_b.mean()))/scale,
            "spectral_js_divergence": spectral})
    cross_errors = []
    for first in range(real.shape[2]):
        for second in range(first+1, real.shape[2]):
            for lag in (0, 1):
                def correlation(values):
                    a = values[:, :values.shape[1]-lag if lag else None, first].ravel()
                    b = values[:, lag:, second].ravel()
                    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else None
                a, b = correlation(real), correlation(synthetic)
                if a is not None and b is not None:
                    cross_errors.append(abs(a-b))
    unique = len({row.tobytes() for row in np.ascontiguousarray(synthetic)})/len(synthetic)
    return {"per_feature": per_feature, "cross_correlation_error": float(np.mean(cross_errors)) if cross_errors else None,
            "unique_sequence_ratio": unique,
            "limitations": "Fixed-window descriptive metrics; overlapping windows are dependent. KS is a statistic, not a significance claim. ACF/spectra/cross-correlation return null when undefined; seasonality requires adequate regular observations."}


def distance_privacy(train, holdout, synthetic, seed=42, cap=128, numerical=None):
    """Train/holdout control for a simple released-synthetic nearest-distance attack."""
    rng = np.random.default_rng(seed)
    if isinstance(train, pd.DataFrame):
        columns = list(train)
        numerical = [c for c in columns if pd.api.types.is_numeric_dtype(train[c])] if numerical is None else numerical
        def sample(frame):
            return frame.sample(n=min(len(frame), cap), random_state=seed).reset_index(drop=True)
        real, held, generated = sample(train), sample(holdout), sample(synthetic)
        def distances(left, right):
            matrix = np.zeros((len(left), len(right)))
            for column in columns:
                if column in numerical:
                    a = pd.to_numeric(left[column]).to_numpy(dtype=float, na_value=np.nan)
                    b = pd.to_numeric(right[column]).to_numpy(dtype=float, na_value=np.nan)
                    scale = float(pd.to_numeric(train[column]).std(ddof=0))
                    scale = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
                    part = np.abs(a[:, None]-b[None, :])/scale
                    part = np.where(np.isnan(a[:, None]) & np.isnan(b[None, :]), 0,
                                    np.where(np.isnan(part), 1, part))
                else:
                    part = (left[column].astype(str).to_numpy()[:, None] != right[column].astype(str).to_numpy()[None, :]).astype(float)
                matrix += part / len(columns)
            return matrix
        exact = float(pd.util.hash_pandas_object(synthetic, index=False).isin(pd.util.hash_pandas_object(train, index=False)).mean())
    else:
        def sample(values):
            return values[rng.choice(len(values), min(cap, len(values)), replace=False)].reshape(min(cap, len(values)), -1)
        real, held, generated = sample(train), sample(holdout), sample(synthetic)
        scale = np.where(real.std(0) > 1e-8, real.std(0), 1.0)
        def distances(left, right):
            return cdist(left/scale, right/scale)/np.sqrt(real.shape[1])
        original = {row.tobytes() for row in np.ascontiguousarray(train)}
        exact = float(np.mean([row.tobytes() in original for row in np.ascontiguousarray(synthetic)]))
    to_train, to_holdout = distances(generated, real).min(1), distances(generated, held).min(1)
    member_distance = distances(real, generated).min(1)
    nonmember_distance = distances(held, generated).min(1)
    attack_auc = float(roc_auc_score(np.r_[np.ones(len(real)), np.zeros(len(held))], -np.r_[member_distance, nonmember_distance]))
    return {"exact_training_match_rate": exact, "near_training_match_rate": float(np.mean(to_train < 0.01)),
        "median_distance_to_train": float(np.median(to_train)), "median_distance_to_holdout": float(np.median(to_holdout)),
        "distance_attack_auc": attack_auc, "sample_cap": cap, "train_sample_count": len(real), "holdout_sample_count": len(held),
        "limitations": "Empirical, sampled distance attack with fixed orientation; not a formal privacy guarantee. Compare against real holdout controls. Values are not confidence bounds."}


def forecasting_utility(train, test, synthetic, target_index, horizon, span):
    if horizon < 1 or horizon >= train.shape[1]:
        raise ValueError("Forecast horizon must be smaller than the sequence length.")
    def xy(windows):
        return windows[:, :-horizon].reshape(len(windows), -1), windows[:, -horizon:, target_index]
    test_x, test_y = xy(test)
    metrics = {}
    for name, values in (("real_only", train), ("synthetic_only", synthetic), ("augmented", np.concatenate([train, synthetic]))):
        x, y = xy(values)
        model = Ridge(alpha=1.0).fit(x, y)
        prediction = model.predict(test_x).reshape(test_y.shape)
        metrics[name] = {"mae": float(mean_absolute_error(test_y, prediction)*span/2),
                         "rmse": float(np.sqrt(mean_squared_error(test_y, prediction))*span/2)}
    return {"horizon": horizon, "target_index": target_index, "model": "Ridge(alpha=1)",
            "metrics": metrics, "units": "original target units", "evaluation_role": "chronological_final_test"}
