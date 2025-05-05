import pandas as pd
import numpy as np
from ta import add_all_ta_features
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import yfinance as yf

# ディレクトリ作成
os.makedirs("model/LightGBM_20250504", exist_ok=True)
os.makedirs("model/LSTM_20250504", exist_ok=True)
os.makedirs("output", exist_ok=True)

# データ取得（yfinance）
ticker = "7203.T"
data = yf.Ticker(ticker).history(period="60d", interval="5m")
data = data[["Open", "High", "Low", "Close", "Volume"]]
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# 特徴量生成
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
print("Infinite values after TA:", np.isinf(data).sum().sum())
print("Columns with inf:", np.isinf(data).sum()[np.isinf(data).sum() > 0])
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# ラグ特徴量
lag_features = {}
for lag in [1, 5, 10]:
    lag_features[f"Close_lag_{lag}"] = data["Close"].shift(lag)
    lag_features[f"Volume_lag_{lag}"] = data["Volume"].shift(lag)
    lag_features[f"Close_pct_change_{lag}"] = data["Close"].pct_change(periods=lag)
lag_df = pd.DataFrame(lag_features, index=data.index)
data = pd.concat([data, lag_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# 統計量
stat_features = {}
for window in [5, 10, 20]:
    stat_features[f"Close_mean_{window}"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}"] = data["Volume"].rolling(window=window).mean()
stat_df = pd.DataFrame(stat_features, index=data.index)
data = pd.concat([data, stat_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# 寄り/引け特徴量
open_close_features = {
    "Open_to_prev_close": data["Open"] - data["Close"].shift(1),
    "Open_to_sma5": data["Open"] - data["trend_sma_fast"],
    "Open_volatility": data["Open"].rolling(window=5).std(),
    "Close_to_sma5": data["Close"] - data["trend_sma_fast"],
    "Close_to_vwap": data["Close"] - data["volume_vwap"],
    "Close_volatility": data["Close"].rolling(window=5).std()
}
open_close_df = pd.DataFrame(open_close_features, index=data.index)
data = pd.concat([data, open_close_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# 特徴量合成
synth_features = {
    "Close_lag_1_plus_Volume_lag_1": np.clip(data["Close_lag_1"] + data["Volume_lag_1"], -1e6, 1e6),
    "Close_lag_1_div_Volume_lag_1": np.clip(data["Close_lag_1"] / data["Volume_lag_1"].replace(0, np.nan), -1e6, 1e6),
    "RSI_minus_MACD": np.clip(data["momentum_rsi"] - data["trend_macd"], -1e6, 1e6),
    "Close_mean_5_plus_BBW": np.clip(data["Close_mean_5"] + data["volatility_bbw"], -1e6, 1e6),
    "Close_lag_5_div_SMA5": np.clip(data["Close_lag_5"] / data["trend_sma_fast"].replace(0, np.nan), -1e6, 1e6),
    "Close_lag_1_times_Volume_lag_1": np.clip(data["Close_lag_1"] * data["Volume_lag_1"], -1e6, 1e6)
}
synth_df = pd.DataFrame(synth_features, index=data.index)
data = pd.concat([data, synth_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# 乖離/インバランス
imbalance_features = {
    "Close_Open_diff": data["Close"] - data["Open"],
    "High_Low_diff": data["High"] - data["Low"],
    "Close_SMA5_diff": data["Close"] - data["trend_sma_fast"],
    "Buy_Sell_Imbalance": np.clip((data["Close"] - data["Open"]) / (data["High"] - data["Low"]).replace(0, np.nan), -1e6, 1e6),
    "Volume_Imbalance": np.clip(data["Volume"] / data["Volume_mean_5"], -1e6, 1e6),
    "RSI_MACD_Imbalance": np.clip(data["momentum_rsi"] / data["trend_macd"].replace(0, np.nan), -1e6, 1e6)
}
imbalance_df = pd.DataFrame(imbalance_features, index=data.index)
data = pd.concat([data, imbalance_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# カテゴリカル変数
cat_features = {
    "hour": data.index.hour,
    "day_of_week": data.index.dayofweek,
    "is_opening": ((data.index.hour == 9) & (data.index.minute <= 30)).astype(int),
    "is_closing": ((data.index.hour == 14) & (data.index.minute >= 30)).astype(int)
}
cat_df = pd.DataFrame(cat_features, index=data.index)
data = pd.concat([data, cat_df], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# センチメント（ダミー）
data["sentiment"] = np.random.uniform(-1, 1, len(data))

# 無限大チェック
print("Infinite values in data:", np.isinf(data).sum().sum())
print("Columns with inf:", np.isinf(data).sum()[np.isinf(data).sum() > 0])
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# 特徴量リスト
columns_to_drop = ["Close", "Open", "High", "Low", "Volume"]
features = data.drop(columns=[col for col in columns_to_drop if col in data.columns]).columns
print(f"Total features: {len(features)}")

# ターゲット（i+1～i+5の終値）
y = pd.DataFrame({
    f"Close_i+{k}": data["Close"].shift(-k) for k in range(1, 6)
}).dropna()
X = data[features].loc[y.index].fillna(data.mean())

# カテゴリカル特徴量を分離
cat_columns = ["hour", "day_of_week", "is_opening", "is_closing"]
X_cat = X[cat_columns]
X_num = X.drop(columns=cat_columns)

# 特徴量の正規化（数値特徴量のみ）
scaler_X = StandardScaler()
index = X_num.index
columns = X_num.columns
X_num = np.clip(X_num, -1e6, 1e6)
X_num = np.nan_to_num(X_num, nan=0, posinf=1e6, neginf=-1e6)
X_num_scaled = scaler_X.fit_transform(X_num)
X_num = pd.DataFrame(X_num_scaled, index=index, columns=columns)

# カテゴリカル特徴量を結合
X = pd.concat([X_num, X_cat], axis=1)

# ターゲットの正規化（列ごとに）
scalers_y = {k: StandardScaler() for k in range(1, 6)}
y_scaled = y.copy()
for k in range(1, 6):
    col = f"Close_i+{k}"
    y_scaled[col] = scalers_y[k].fit_transform(y[[col]])
y = y_scaled

# カテゴリカル変数の処理（LSTM用）
X_lstm = pd.get_dummies(X, columns=cat_columns, drop_first=True)
columns_lstm = X_lstm.columns
X_lstm = np.nan_to_num(X_lstm, nan=0, posinf=1e6, neginf=-1e6)
features_lstm = columns_lstm

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
metrics = []

# LightGBM
for k in range(1, 6):
    rmses, maes = [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[f"Close_i+{k}"].iloc[train_idx], y[f"Close_i+{k}"].iloc[test_idx]
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_columns)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "random_state": 42,
            "verbose": -1
        }
        model = lgb.train(params, train_data, valid_sets=[test_data], callbacks=[lgb.early_stopping(50, verbose=True)])
        joblib.dump(model, f"model/LightGBM_20250504/model_i+{k}.joblib")
        y_pred = model.predict(X_test)
        y_pred_scaled = scalers_y[k].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_pred_orig = np.clip(y_pred_scaled, 2000, 3000)
        y_test_orig = scalers_y[k].inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        rmses.append(mean_squared_error(y_test_orig, y_pred_orig, squared=False))
        maes.append(mean_absolute_error(y_test_orig, y_pred_orig))
        if k == 1:
            importances = model.feature_importance(importance_type="gain")
            feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(by="importance", ascending=False)
            plt.figure(figsize=(10, 8))
            sns.barplot(x="importance", y="feature", data=feature_importance_df.head(30), palette="viridis")
            plt.title("Top 30 Feature Importance (LightGBM)")
            plt.savefig("output/feature_importance_lgb.png")
            plt.close()
    metrics.append({"Model": f"LightGBM_i+{k}", "RMSE": np.mean(rmses), "MAE": np.mean(maes)})

# LSTM（5本分の終値をまとめて予測）
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
# y_lstmを(サンプル数, 5)に整形
y_lstm = np.array([y_scaled.iloc[i][[f"Close_i+{k}" for k in range(1, 6)]].values for i in range(timesteps, len(y))])
X_lstm_3d = np.nan_to_num(X_lstm_3d, nan=0, posinf=1e6, neginf=-1e6)
lstm_metrics = []

# 1つのLSTMモデルで5本分をまとめて予測
rmses, maes = [], []
for train_idx, test_idx in tscv.split(X_lstm_3d):
    X_train, X_test = X_lstm_3d[train_idx], X_lstm_3d[test_idx]
    y_train, y_test = y_lstm[train_idx], y_lstm[test_idx]
    model = create_lstm_model((timesteps, len(features_lstm)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    model.save(f"model/LSTM_20250504/model_multi.h5")
    y_pred = model.predict(X_test, verbose=0)
    # 各タイムステップごとに逆変換とクリッピング
    y_pred_orig = np.zeros_like(y_pred)
    y_test_orig = np.zeros_like(y_test)
    for k in range(5):
        y_pred_orig[:, k] = np.clip(scalers_y[k+1].inverse_transform(y_pred[:, k].reshape(-1, 1)).flatten(), 2000, 3000)
        y_test_orig[:, k] = scalers_y[k+1].inverse_transform(y_test[:, k].reshape(-1, 1)).flatten()
    # 各タイムステップごとの評価
    for k in range(5):
        rmse_k = mean_squared_error(y_test_orig[:, k], y_pred_orig[:, k], squared=False)
        mae_k = mean_absolute_error(y_test_orig[:, k], y_pred_orig[:, k])
        if k not in lstm_metrics:
            lstm_metrics.append({"Model": f"LSTM_i+{k+1}", "RMSE": [rmse_k], "MAE": [mae_k]})
        else:
            lstm_metrics[k]["RMSE"].append(rmse_k)
            lstm_metrics[k]["MAE"].append(mae_k)
    # 特徴量重要度の簡易推定（k=1のみ）
    baseline_rmse = mean_squared_error(y_test_orig[:, 0], y_pred_orig[:, 0], squared=False)
    importance_dict = {}
    for i in range(len(features_lstm)):
        X_test_temp = X_test.copy()
        X_test_temp[:, :, i] = 0
        y_pred_temp = model.predict(X_test_temp, verbose=0)
        y_pred_temp_orig = np.clip(scalers_y[1].inverse_transform(y_pred_temp[:, 0].reshape(-1, 1)).flatten(), 2000, 3000)
        temp_rmse = mean_squared_error(y_test_orig[:, 0], y_pred_temp_orig, squared=False)
        importance_dict[features_lstm[i]] = baseline_rmse - temp_rmse
    feature_importance_df = pd.DataFrame({"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())}).sort_values(by="importance", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=feature_importance_df.head(30), palette="viridis")
    plt.title("Top 30 Feature Importance (LSTM)")
    plt.savefig("output/feature_importance_lstm.png")
    plt.close()

# LSTMメトリクスの平均を計算
for k in range(5):
    lstm_metrics[k]["RMSE"] = np.mean(lstm_metrics[k]["RMSE"])
    lstm_metrics[k]["MAE"] = np.mean(lstm_metrics[k]["MAE"])

# メトリクス保存
metrics_df = pd.DataFrame(metrics + lstm_metrics)
metrics_df.to_csv("output/metrics.csv", index=False)

# 可視化
lstm_model = tf.keras.models.load_model(f"model/LSTM_20250504/model_multi.h5")
y_pred_lstm = lstm_model.predict(X_lstm_3d, verbose=0)
y_pred_lstm_orig = np.zeros((y_pred_lstm.shape[0], 5))
y_orig = np.zeros((y_lstm.shape[0], 5))
for k in range(5):
    y_pred_lstm_orig[:, k] = np.clip(scalers_y[k+1].inverse_transform(y_pred_lstm[:, k].reshape(-1, 1)).flatten(), 2000, 3000)
    y_orig[:, k] = scalers_y[k+1].inverse_transform(y_lstm[:, k].reshape(-1, 1)).flatten()
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

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="RMSE", data=metrics_df)
plt.title("RMSE Comparison")
plt.xticks(rotation=45)
plt.savefig("output/rmse_comparison.png")
plt.close()