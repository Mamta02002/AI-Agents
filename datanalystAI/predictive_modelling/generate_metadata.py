import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
from sklearn.utils import resample

def is_time_series(df):
    return isinstance(df.index, pd.DatetimeIndex)

def is_sequential(df):
    return df.index.is_monotonic_increasing and df.index.is_numeric()

def has_missing_values(df):
    return df.isnull().any().any()

def is_imbalanced(y, threshold=0.75):
    if y.dtype == 'object' or y.nunique() < 20:
        counts = y.value_counts(normalize=True)
        return counts.max() > threshold
    return False

def features_independent(df):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_triangle.max().max() < 0.3

def requires_interpretability(df):
    return df.shape[1] < 10  # fewer features often implies desire for interpretability

def has_complex_boundaries(df, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    if y.dtype == 'object' or y.nunique() < 20:
        rf = RandomForestClassifier()
        score = cross_val_score(rf, df, y, cv=3)
        return score.std() > 0.05  # high variance implies complex decision boundaries
    return False

def is_tabular(df):
    return all(df.dtypes.apply(lambda dt: dt.kind in 'biufc'))  # numeric or encoded

def is_univariate(df, target=None):
    if target:
        return df.drop(columns=[target]).shape[1] == 0
    else:
        return df.shape[1]==0

def detect_autocorrelation(y, threshold=0.5, lags=20):
    autocorr_values = acf(y, nlags=lags, fft=True)
    return np.any(np.abs(autocorr_values[1:]) > threshold)

def detect_seasonality(y, freq_threshold=0.1):
    freqs, power = periodogram(y)
    dominant_freq = freqs[np.argmax(power[1:]) + 1]
    return dominant_freq > freq_threshold

def generate_metadata(df, target=None):
    if target:
        y = df[target]
        X = df.drop(columns=[target])

    metadata = {
        'is_time_series': is_time_series(df),
        'is_sequential': is_sequential(df),
        'has_missing_values': has_missing_values(df),
        'is_imbalanced': is_imbalanced(y),
        'features_independent': features_independent(X),
        'requires_interpretability': requires_interpretability(X),
        'has_complex_boundaries': has_complex_boundaries(X, y),
        'is_tabular': is_tabular(X),
        'is_univariate': is_univariate(df, target)
    }

    # Optional: add autocorrelation and seasonality flags
    if metadata['is_time_series']:
        metadata['has_autocorrelation'] = detect_autocorrelation(y)
        metadata['has_seasonality'] = detect_seasonality(y)

    return metadata
