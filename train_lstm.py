import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import os

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X_lstm = data_dict["X_lstm"]
y = data_dict["y"]
scalers_y = data_dict["scalers_y"]
features_lstm = data_dict["features_lstm"]

# ディレクトリ作成
os.makedirs("model/LSTM_20250504", exist_ok=True)

# LSTMデータ準備
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(5)  # 5本分の終値を予測
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

timesteps = 50
X_lstm_3d = np.array([X_lstm[i-timesteps:i] for i in range(timesteps, len(X_lstm))])
y_lstm = np.array([y.iloc[i][[f"Close_i+{k}" for k in range(1, 6)]].values for i in range(timesteps, len(y))])
X_lstm_3d = np.nan_to_num(X_lstm_3d, nan=0, posinf=1e6, neginf=-1e6)

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
lstm_metrics = []

# LSTM学習
rmses, maes = [], []
for train_idx, test_idx in tscv.split(X_lstm_3d):
    X_train, X_test = X_lstm_3d[train_idx], X_lstm_3d[test_idx]
    y_train, y_test = y_lstm[train_idx], y_lstm[test_idx]
    model = create_lstm_model((timesteps, len(features_lstm)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    model.save(f"model/LSTM_20250504/model_multi.h5")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_orig = np.zeros_like(y_pred)
    y_test_orig = np.zeros_like(y_test)
    for k in range(5):
        y_pred_orig[:, k] = np.clip(scalers_y[k+1].inverse_transform(y_pred[:, k].reshape(-1, 1)).flatten(), 2000, 3000)
        y_test_orig[:, k] = scalers_y[k+1].inverse_transform(y_test[:, k].reshape(-1, 1)).flatten()
    for k in range(5):
        rmse_k = mean_squared_error(y_test_orig[:, k], y_pred_orig[:, k], squared=False)
        mae_k = mean_absolute_error(y_test_orig[:, k], y_pred_orig[:, k])
        if k not in lstm_metrics:
            lstm_metrics.append({"Model": f"LSTM_i+{k+1}", "RMSE": [rmse_k], "MAE": [mae_k]})
        else:
            lstm_metrics[k]["RMSE"].append(rmse_k)
            lstm_metrics[k]["MAE"].append(mae_k)

# LSTMメトリクスの平均を計算
for k in range(5):
    lstm_metrics[k]["RMSE"] = np.mean(lstm_metrics[k]["RMSE"])
    lstm_metrics[k]["MAE"] = np.mean(lstm_metrics[k]["MAE"])

# メトリクス保存
metrics_df = pd.DataFrame(lstm_metrics)
metrics_df.to_csv("output/lstm_metrics.csv", index=False)