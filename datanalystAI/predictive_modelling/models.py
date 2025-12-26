from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, LogisticRegression

def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return model, score

def train_xgboost_classifier(X_train, y_train, X_test, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return model, score

def train_xgboost_regressor(X_train, y_train, X_test, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return model, score

def train_lstm(X_train, y_train, X_test, y_test, epochs=10):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_seq = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train, epochs=epochs, verbose=0)

    y_pred = model.predict(X_test_seq).flatten()
    score = r2_score(y_test, y_pred)
    return model, score

def train_arima(y_train, y_test, order=(1,1,1)):
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    y_pred = model_fit.forecast(steps=len(y_test))
    score = 1 - mean_squared_error(y_test, y_pred) / np.var(y_test)
    return model_fit, score

def train_sarima(y_train, y_test, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    y_pred = model_fit.forecast(steps=len(y_test))
    score = 1 - mean_squared_error(y_test, y_pred) / np.var(y_test)
    return model_fit, score

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return model, score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return model, score

def get_model(algo, x_train, y_train, x_test, y_test):
    algo = algo.lower()
    model_fit, accuracy, importance = None, None, None
    if algo == 'naivebayes':
        model_fit, accuracy = train_naive_bayes(x_train, y_train, x_test, y_test)
        importance = model_fit.theta_[0]  # Mean log likelihoods for each feature
    elif algo == 'logisticregression':
        model_fit, accuracy = train_linear_regression(x_train, y_train, x_test, y_test)
        importance = model_fit.coef_
        if importance.ndim > 1:  # Multiclass logistic regression
            importance = importance[0]
    elif algo == 'linearregression':
        model_fit, accuracy = train_logistic_regression(x_train, y_train, x_test, y_test)
        importance = model_fit.coef_
        if importance.ndim > 1:  # Multiclass logistic regression
            importance = importance[0]
    elif algo == 'arima':
        model_fit, accuracy = train_arima(y_train, y_test)
    elif algo == 'sarima':
        model_fit, accuracy = train_sarima(y_train, y_test)
    elif algo == 'xgboostregressor':
        model_fit, accuracy = train_xgboost_regressor(x_train, y_train, x_test, y_test)
        importance = model_fit.feature_importances_
    elif algo == 'xgboostclassifier':
        model_fit, accuracy = train_xgboost_classifier(x_train, y_train, x_test, y_test)
        importance = model_fit.feature_importances_
    elif algo == 'lstm':
        model_fit, accuracy = train_lstm(x_train, y_train, x_test, y_test)
    return model_fit, accuracy, importance