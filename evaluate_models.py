import pandas as pd
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X = data_dict["X"]
X_lstm = data_dict["X_lstm"]
y = data_dict["y"]
scalers_y = data_dict["scalers_y"]
features_lstm = data_dict["features_lstm"]

# ディレクトリ作成
os.makedirs("output", exist_ok=True)

# LSTMデータ準備
timesteps = 50
X_lstm_3d = np.array([X_lstm[i-timesteps:i] for i in range(timesteps, len(X_lstm))])
y_lstm = np.array([y.iloc[i][[f"Close_i+{k}" for k in range(1, 6)]].values for i in range(timesteps, len(y))])
y_orig = np.zeros((y_lstm.shape[0], 5))
for k in range(5):
    y_orig[:, k] = scalers_y[k+1].inverse_transform(y_lstm[:, k].reshape(-1, 1)).flatten()

# モデル評価と可視化
lstm_model = tf.keras.models.load_model("model/LSTM_20250504/model_multi.h5")
y_pred_lstm = lstm_model.predict(X_lstm_3d, verbose=0)
y_pred_lstm_orig = np.zeros((y_pred_lstm.shape[0], 5))
for k in range(5):
    y_pred_lstm_orig[:, k] = np.clip(scalers_y[k+1].inverse_transform(y_pred_lstm[:, k].reshape(-1, 1)).flatten(), 2000, 3000)

for k in range(1, 6):
    lgb_model = joblib.load(f"model/LightGBM_20250504/model_i+{k}.joblib")
    y_pred_lgb = lgb_model.predict(X)
    y_pred_lgb_scaled = scalers_y[k].inverse_transform(y_pred_lgb.reshape(-1, 1)).flatten()
    y_pred_lgb_orig = np.clip(y_pred_lgb_scaled, 2000, 3000)
    plt.figure(figsize=(12, 6))
    plt.plot(y.index[-100:], y_orig[-100:, k-1], label="Actual", color="blue")
    plt.plot(y.index[-100:], y_pred_lgb_orig[-100:], label="LightGBM", color="red", linestyle="--")
    plt.plot(y.index[-100:], y_pred_lstm_orig[-100:, k-1], label="LSTM", color="green", linestyle="-.")
    plt.title(f"Prediction for Close_i+{k}")
    plt.xlabel("Date")
    plt.ylabel("Close Price (JPY)")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/timeseries_i+{k}.png")
    plt.close()
    errors_lgb = y_orig[-100:, k-1] - y_pred_lgb_orig[-100:]
    errors_lstm = y_orig[-100:, k-1] - y_pred_lstm_orig[-100:, k-1]
    plt.figure(figsize=(8, 6))
    sns.histplot(errors_lgb, bins=50, kde=True, color="purple", label="LightGBM")
    sns.histplot(errors_lstm, bins=50, kde=True, color="green", label="LSTM", alpha=0.5)
    plt.title(f"Error Distribution for Close_i+{k}")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/error_dist_i+{k}.png")
    plt.close()

# RMSE比較
lightgbm_metrics = pd.read_csv("output/lightgbm_metrics.csv")
lstm_metrics = pd.read_csv("output/lstm_metrics.csv")
metrics_df = pd.concat([lightgbm_metrics, lstm_metrics], ignore_index=True)
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="RMSE", data=metrics_df)
plt.title("RMSE Comparison")
plt.xticks(rotation=45)
plt.savefig("output/rmse_comparison.png")
plt.close()