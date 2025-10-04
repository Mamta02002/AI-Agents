from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_column=None, scale_numeric=True, encode_categorical=True):
        self.target_column = target_column
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical
        self.label_encoder = None
        self.onehot_encoder = None
        self.scaler = None
        self.numeric_columns = []
        self.categorical_columns = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()

        # Identify column types
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target column from categorical list if present
        if self.target_column in self.categorical_columns:
            self.categorical_columns.remove(self.target_column)

        # Fit label encoder for target column
        if self.target_column and self.target_column in df.columns:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df[self.target_column])

        # Fit one-hot encoder for categorical columns
        if self.encode_categorical and self.categorical_columns:
            self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.onehot_encoder.fit(df[self.categorical_columns])

        # Fit scaler for numeric columns
        if self.scale_numeric and self.numeric_columns:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numeric_columns])

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # Encode target column
        if self.target_column and self.target_column in df.columns:
            df[self.target_column] = self.label_encoder.transform(df[self.target_column])

        # One-hot encode categorical columns
        if self.encode_categorical and self.categorical_columns:
            cat_encoded = self.onehot_encoder.transform(df[self.categorical_columns])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=self.onehot_encoder.get_feature_names_out(self.categorical_columns), index=df.index)
            df.drop(columns=self.categorical_columns, inplace=True)
            df = pd.concat([df, cat_encoded_df], axis=1)

        # Scale numeric columns
        if self.scale_numeric and self.numeric_columns:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        return df

def encode_data(df, target):
    pipeline = Pipeline([
        ('encoder', DataEncoder(target_column=target))
    ])
    return pipeline.fit_transform(df)
