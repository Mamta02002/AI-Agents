from predictive_modelling.models import get_model
from predictive_modelling.generate_metadata import generate_metadata
from preprocessing.encoder import encode_data
from preprocessing.process_data import transform_data
import numpy as np
from sklearn.model_selection import train_test_split


def select_algorithm(problem_type, metadata):
    """
    Enhanced selector with autocorrelation and seasonality awareness.
    """
    if problem_type == 'classification':
        if metadata['features_independent'] and metadata['is_tabular']:
            return 'NaiveBayes'
        elif metadata['requires_interpretability']:
            return 'LogisticRegression'
        # elif metadata['has_complex_boundaries'] and X.shape[0] < 1000:
        #     return 'SVC'
        else:
            return 'XGBoostClassifier'

    elif problem_type == 'regression':
        if metadata['is_time_series']:
            if metadata['is_univariate']:
                if metadata['has_seasonality']:
                    return 'SARIMA'
                elif metadata['has_autocorrelation']:
                    return 'ARIMA'
                else:
                    return 'XGBoostRegressor'
            else:
                return 'SARIMA' if metadata['has_seasonality'] else 'XGBoostRegressor'

        elif metadata['is_sequential']:
            return 'LSTM'
        elif metadata['requires_interpretability']:
            return 'LinearRegression'
        else:
            return 'XGBoostRegressor'

    else:
        return 'Unsupported problem type'

def detect_problem_type(y):
    if y.dtype == 'object' or y.nunique() < 20:
        return 'classification'
    elif np.issubdtype(y.dtype, np.number):
        return 'regression'
    else:
        return 
    
def train_model(data, target=None):
    model_fit, accuracy, importance = None, None, None
    if target:
        problem_type = detect_problem_type(data[target])
        # print("problem type", problem_type)
        transformed_data = transform_data(data, target)
        # print(transformed_data.columns)
        encoded_data = encode_data(transformed_data, target)
        metadata = generate_metadata(encoded_data, target)
        algo = select_algorithm(problem_type, metadata)
        x_train, x_test, y_train, y_test = train_test_split(encoded_data.drop([target], axis=1), encoded_data[target], test_size=0.2, random_state=42)
        model_fit, accuracy, importance = get_model(algo, x_train, y_train, x_test, y_test)
    else:
        print("No target column specified for predictive modelling")
    return model_fit, accuracy, importance
