from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import entropy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class EnhancedDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 target_column=None):
        self.target_column = target_column
        self.columns_to_keep = None
        self.impute_values = {}

    def fit(self, X, y=None):
        df = X.copy()
        df.columns = [str(col).strip().capitalize() for col in df.columns]
        df = df.drop_duplicates()
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        df = df.loc[:, df.nunique() > 1]
        self.columns_to_keep = df.columns.tolist()

        for col in df.columns:
            if col == self.target_column:
                continue

            if df[col].isnull().any():
                if np.issubdtype(df[col].dtype, np.number):
                    skewness = df[col].skew()
                    if abs(skewness) > 1:
                        self.impute_values[col] = df[col].median()
                    else:
                        self.impute_values[col] = df[col].mean()
                
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    self.impute_values[col] = 'ffill'
                
                elif df[col].dtype == 'bool':
                    self.impute_values[col] = df[col].mode()[0]
                elif df[col].dtype == 'object':
                    self.impute_values[col] = df[col].mode()[0]
                else:
                    self.impute_values[col] = "Missing"

        return self

    def transform(self, X):
        df = X.copy()
        df.columns = [str(col).strip().capitalize() for col in df.columns]
        df = df[self.columns_to_keep]
        for col, value in self.impute_values.items():
                if value == 'ffill':
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    df[col].fillna(value, inplace=True)
        return df


class EnhancedDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variance_threshold=0.0,
                 entropy_threshold=0.0,
                 corr_threshold=0.9,
                 target_column=None,
                 dependency_threshold=0.01):
        self.variance_threshold = variance_threshold
        self.entropy_threshold = entropy_threshold
        self.corr_threshold = corr_threshold
        self.target_column = target_column.strip().lower().replace(' ', '_')
        self.dependency_threshold = dependency_threshold
        self.columns_to_keep = None

    def _filter_by_target_dependency(self, df, y):
        selected_cols = []
        for col in df.columns:
            if col == self.target_column:
                continue
            if df[col].dtype in [np.float64, np.int64]:
                y_encoded = LabelEncoder().fit_transform(y)
                corr = np.corrcoef(df[col], y_encoded)[0, 1]
                if not np.isnan(corr) and abs(corr) > self.dependency_threshold:
                    selected_cols.append(col)
            else:
                try:
                    mi = mutual_info_classif(df[[col]], y, discrete_features=True)
                    if mi[0] > self.dependency_threshold:
                        selected_cols.append(col)
                except Exception:
                    pass
        return selected_cols

    def fit(self, X, y=None):
        df = X.copy()
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]

        numeric_df = df.select_dtypes(include=[np.number])
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(numeric_df)
        var_cols = numeric_df.columns[vt.get_support()].tolist()

        ent_cols = []
        for col in df.columns:
            counts = df[col].value_counts()
            ent = entropy(counts)
            if ent > self.entropy_threshold:
                ent_cols.append(col)

        corr_df = df[var_cols].copy()
        corr_matrix = corr_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        if self.target_column and self.target_column in df.columns:
            to_drop = [col for col in to_drop if col != self.target_column]
        corr_cols = [col for col in corr_df.columns if col not in to_drop]

        self.columns_to_keep = list(set(df.columns.tolist()) & set(var_cols) & set(ent_cols) & set(corr_cols))
        
        # Filter by correlation or mutual information with target
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column]
            dependency_cols = self._filter_by_target_dependency(df, y)
            self.columns_to_keep = list(set(self.columns_to_keep) & set(dependency_cols))
            self.columns_to_keep.append(self.target_column)

        return self

    def transform(self, X):
        df = X.copy()
        return df[[col for col in df.columns if str(col).strip().lower().replace(' ', '_') in self.columns_to_keep]]


def preprocess_data(df, target):
    cleaner = EnhancedDataPreprocessor(
        target_column=target
    )

    return cleaner.fit_transform(df)


def transform_data(df, target):
    cleaner = EnhancedDataTransformer(
        variance_threshold=0.01,
        entropy_threshold=0.5,
        corr_threshold=0.85,
        dependency_threshold=0.01,
        target_column=target
    )

    df = cleaner.fit_transform(df)
    return df

