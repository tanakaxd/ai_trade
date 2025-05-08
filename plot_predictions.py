import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import joblib

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X = data_dict["X"]
y_orig = data_dict["y_orig"]["Close_i+5"]
features = data_dict["features"]

# 重要度1未満の特徴量を除外（train_lightgbm.pyと一致させる）
drop_features = ["Close_SMA5_diff", "Close_lag_1_plus_Volume_lag_1", "Close_mean_5", "Close_mean_20", 
                 "hour", "day_of_week", "is_opening", "is_closing", "Close_volatility",
                 "sentiment", "Volume_lag_1", "Close_lag_1_times_Volume_lag_1", "Close_lag_1_div_Volume_lag_1",
                 "Volume_lag_3", "Close_lag_3_div_Volume_lag_3", "High_Low_diff", "Close_lag_5_div_SMA5",
                 "Volume_lag_5", "Buy_Sell_Imbalance", "BBW_times_VWAP", "Volume_Imbalance", "Open_to_prev_close",
                 "Volume_lag_15", "Close_pct_change_15", "Volume_lag_30", "Open_to_sma5", "Close_Open_diff",
                 "Open_volatility", "Volume_lag_20", "Close_lag_10", "Volume_lag_10"]
selected_features = [f for f in features if f not in drop_features]
# 追加特徴量（train_lightgbm.pyで使用）を明示的に追加
selected_features.extend(["sma_200", "atr", "bb_width"])
X = X[selected_features]

# ターゲットの対数変換とスケーリング（train_lightgbm.pyと一致させる）
y_log = np.log1p(y_orig)
scaler_y = RobustScaler()
y = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()
print(f"Scaled y range: min={y.min()}, max={y.max()}")

# モデル読み込み
model = joblib.load("model/LightGBM_20250504/model_i+5.joblib")

# フォールドごとのテストデータを特定
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    y_test_orig = y_orig.iloc[test_idx]

    # 予測
    y_pred = model.predict(X_test)
    print(f"Fold {fold} - Raw predicted y range: min={y_pred.min()}, max={y_pred.max()}")

    # スケール戻しと指数変換
    y_pred_log = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    print(f"Fold {fold} - Inverse transformed y_pred_log range: min={y_pred_log.min()}, max={y_pred_log.max()}")
    y_pred_orig = np.expm1(y_pred_log)
    print(f"Fold {fold} - Final predicted y_orig range: min={y_pred_orig.min()}, max={y_pred_orig.max()}")
    y_pred_orig = np.clip(y_pred_orig, 300, 3500)  # 株価範囲を300〜3500に調整

    # プロット
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig.index, y_test_orig, label="Actual Close_i+5", color="blue")
    plt.plot(y_test_orig.index, y_pred_orig, label="Predicted Close_i+5", color="orange", linestyle="--")
    plt.title(f"Fold {fold}: Actual vs Predicted Close_i+5")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/fold{fold}_predictions_vs_actual.png")
    plt.close()

    # 誤差のプロット
    errors = y_test_orig - y_pred_orig
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig.index, errors, label="Prediction Error (Actual - Predicted)", color="red")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.title(f"Fold {fold}: Prediction Errors")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/fold{fold}_prediction_errors.png")
    plt.close()