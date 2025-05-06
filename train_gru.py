import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ディレクトリ作成
os.makedirs("model/GRU_20250504", exist_ok=True)
os.makedirs("output", exist_ok=True)

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X_gru = data_dict["X_gru"]
y = data_dict["y"]["Close_i+5"]
y_orig = data_dict["y_orig"]["Close_i+5"]
scaler_y = data_dict["scaler_y"]
features_gru = data_dict["features_gru"]

# GRUデータ準備
def create_gru_model(input_shape):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(1)  # 単一の終値（Close_i+5）を予測
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

timesteps = 50
X_gru_3d = np.array([X_gru[i-timesteps:i] for i in range(timesteps, len(X_gru))])
y_gru = np.array(y.iloc[timesteps:])
X_gru_3d = np.nan_to_num(X_gru_3d, nan=0, posinf=1e6, neginf=-1e6)

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []
average_metrics = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_gru_3d)):
    X_train, X_test = X_gru_3d[train_idx], X_gru_3d[test_idx]
    y_train, y_test = y_gru[train_idx], y_gru[test_idx]
    y_test_orig = y_orig.iloc[timesteps:][test_idx]
    
    # GRU学習
    model = create_gru_model((timesteps, len(features_gru)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    
    # モデル保存
    model.save(f"model/GRU_20250504/model_fold_{fold+1}.h5")
    
    # 評価
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.clip(y_pred_orig, 2000, 3000)
    
    # メトリクス計算
    rmse = mean_squared_error(y_test_orig, y_pred_orig, squared=False)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    
    fold_metrics.append({
        "Fold": fold + 1,
        "Model": "GRU_i+5",
        "RMSE": rmse,
        "MAE": mae
    })

# フォールドごとのメトリクス保存
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_df.to_csv("output/gru_fold_metrics.csv", index=False)

# 平均メトリクス計算
rmses = [m["RMSE"] for m in fold_metrics]
maes = [m["MAE"] for m in fold_metrics]
average_metrics.append({
    "Model": "GRU_i+5",
    "RMSE": np.mean(rmses),
    "MAE": np.mean(maes)
})

# 平均メトリクス保存
average_metrics_df = pd.DataFrame(average_metrics)
average_metrics_df.to_csv("output/gru_metrics.csv", index=False)