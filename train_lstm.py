import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ディレクトリ作成
os.makedirs("model/LSTM_20250504", exist_ok=True)
os.makedirs("output", exist_ok=True)

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X_lstm = data_dict["X_lstm"]
y = data_dict["y"]
scalers_y = data_dict["scalers_y"]
features_lstm = data_dict["features_lstm"]

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
fold_metrics = []  # フォールドごとのメトリクスを保存
average_metrics = []  # 平均メトリクスを保存

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_lstm_3d)):
    X_train, X_test = X_lstm_3d[train_idx], X_lstm_3d[test_idx]
    y_train, y_test = y_lstm[train_idx], y_lstm[test_idx]
    
    # LSTM学習
    model = create_lstm_model((timesteps, len(features_lstm)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    
    # 各フォールドのモデルを保存
    model.save(f"model/LSTM_20250504/model_multi_fold_{fold+1}.h5")
    
    # 評価
    y_pred = model.predict(X_test, verbose=0)
    y_pred_orig = np.zeros_like(y_pred)
    y_test_orig = np.zeros_like(y_test)
    for k in range(5):
        y_pred_orig[:, k] = np.clip(scalers_y[k+1].inverse_transform(y_pred[:, k].reshape(-1, 1)).flatten(), 2000, 3000)
        y_test_orig[:, k] = scalers_y[k+1].inverse_transform(y_test[:, k].reshape(-1, 1)).flatten()
    
    # フォールドごとのメトリクスを計算
    for k in range(5):
        rmse_k = mean_squared_error(y_test_orig[:, k], y_pred_orig[:, k], squared=False)
        mae_k = mean_absolute_error(y_test_orig[:, k], y_pred_orig[:, k])
        fold_metrics.append({
            "Fold": fold + 1,
            "Model": f"LSTM_i+{k+1}",
            "RMSE": rmse_k,
            "MAE": mae_k
        })

# フォールドごとのメトリクスを保存
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_df.to_csv("output/lstm_fold_metrics.csv", index=False)

# 平均メトリクスを計算
for k in range(5):
    model_name = f"LSTM_i+{k+1}"
    fold_rmse = [m["RMSE"] for m in fold_metrics if m["Model"] == model_name]
    fold_mae = [m["MAE"] for m in fold_metrics if m["Model"] == model_name]
    average_metrics.append({
        "Model": model_name,
        "RMSE": np.mean(fold_rmse),
        "MAE": np.mean(fold_mae)
    })

# 平均メトリクスを保存
average_metrics_df = pd.DataFrame(average_metrics)
average_metrics_df.to_csv("output/lstm_metrics.csv", index=False)