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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import requests
import shap

# 環境変数からFMP APIキーを取得
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY environment variable not set")

# ディレクトリ作成
os.makedirs("model/LightGBM_20250504", exist_ok=True)
os.makedirs("model/LSTM_20250504", exist_ok=True)
os.makedirs("output", exist_ok=True)

# データ取得（FMP API）
ticker = "7203.T"
url = f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?apikey={FMP_API_KEY}"
response = requests.get(url)
data = pd.DataFrame(response.json())
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)
data = data[["open", "high", "low", "close", "volume"]].rename(columns={
    "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
})
data = data.sort_index()

# 特徴量生成
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")

# ラグ特徴量
lag_features = {}
for lag in [1, 5, 10]:
    lag_features[f"Close_lag_{lag}"] = data["Close"].shift(lag)
    lag_features[f"Volume_lag_{lag}"] = data["Volume"].shift(lag)
    lag_features[f"Close_pct_change_{lag}"] = data["Close"].pct_change(periods=lag)
data = pd.concat([data, pd.DataFrame(lag_features, index=data.index)], axis=1)

# 統計量
stat_features = {}
for window in [5, 10, 20]:
    stat_features[f"Close_mean_{window}"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}"] = data["Volume"].rolling(window=window).mean()
data = pd.concat([data, pd.DataFrame(stat_features, index=data.index)], axis=1)

# 寄り/引け特徴量
open_close_features = {
    "Open_to_prev_close": data["Open"] - data["Close"].shift(1),
    "Open_to_sma5": data["Open"] - data["trend_sma_fast"],
    "Open_volatility": data["Open"].rolling(window=5).std(),
    "Close_to_sma5": data["Close"] - data["trend_sma_fast"],
    "Close_to_vwap": data["Close"] - data["volume_vwap"],
    "Close_volatility": data["Close"].rolling(window=5).std()
}
data = pd.concat([data, pd.DataFrame(open_close_features, index=data.index)], axis=1)

# 特徴量合成
synth_features = {
    "Close_lag_1_plus_Volume_lag_1": data["Close_lag_1"] + data["Volume_lag_1"],
    "Close_lag_1_div_Volume_lag_1": data["Close_lag_1"] / data["Volume_lag_1"].replace(0, np.nan),
    "RSI_minus_MACD": data["momentum_rsi"] - data["trend_macd"],
    "Close_mean_5_plus_BBW": data["Close_mean_5"] + data["volatility_bbw"],
    "Close_lag_5_div_SMA5": data["Close_lag_5"] / data["trend_sma_fast"].replace(0, np.nan),
    "Close_lag_1_times_Volume_lag_1": data["Close_lag_1"] * data["Volume_lag_1"]  # クロス乗算（オプション）
}
data = pd.concat([data, pd.DataFrame(synth_features, index=data.index)], axis=1)

# 乖離/インバランス
imbalance_features = {
    "Close_Open_diff": data["Close"] - data["Open"],
    "High_Low_diff": data["High"] - data["Low"],
    "Close_SMA5_diff": data["Close"] - data["trend_sma_fast"],
    "Buy_Sell_Imbalance": (data["Close"] - data["Open"]) / (data["High"] - data["Low"]).replace(0, np.nan),
    "Volume_Imbalance": data["Volume"] / data["Volume_mean_5"],
    "RSI_MACD_Imbalance": data["momentum_rsi"] / data["trend_macd"].replace(0, np.nan)
}
data = pd.concat([data, pd.DataFrame(imbalance_features, index=data.index)], axis=1)

# カテゴリカル変数
cat_features = {
    "hour": data.index.hour,
    "day_of_week": data.index.dayofweek,
    "is_opening": ((data.index.hour == 9) & (data.index.minute <= 30)).astype(int),
    "is_closing": ((data.index.hour == 14) & (data.index.minute >= 30)).astype(int)
}
data = pd.concat([data, pd.DataFrame(cat_features, index=data.index)], axis=1)

# センチメント（ダミー）
data["sentiment"] = np.random.uniform(-1, 1, len(data))

# 特徴量リスト
columns_to_drop = ["Close", "Open", "High", "Low", "Volume"]
features = data.drop(columns=[col for col in columns_to_drop if col in data.columns]).columns
print(f"Total features: {len(features)}")

# ターゲット（i+1～i+5の終値）
y = pd.DataFrame({
    f"Close_i+{k}": data["Close"].shift(-k) for k in range(1, 6)
}).dropna()
X = data[features].loc[y.index].fillna(0)

# 特徴量の正規化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# カテゴリカル変数の処理（LSTM用）
X_lstm = pd.get_dummies(X, columns=["hour", "day_of_week", "is_opening", "is_closing"], drop_first=True)
X_lstm = np.nan_to_num(X_lstm, nan=0, posinf=1e6, neginf=-1e6)
features_lstm = X_lstm.columns

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
metrics = []

# LightGBM
for k in range(1, 6):
    rmses, maes = [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[f"Close_i+{k}"].iloc[train_idx], y[f"Close_i+{k}"].iloc[test_idx]
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=["hour", "day_of_week", "is_opening", "is_closing"])
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1
        }
        model = lgb.train(params, train_data, valid_sets=[test_data], callbacks=[lgb.early_stopping(100, verbose=False)])
        joblib.dump(model, f"model/LightGBM_20250504/model_i+{k}.joblib")
        y_pred = model.predict(X_test)
        y_pred = np.nan_to_num(y_pred, nan=0, posinf=1e6, neginf=-1e6)
        rmses.append(mean_squared_error(y_test, y_pred, squared=False))
        maes.append(mean_absolute_error(y_test, y_pred))
        if k == 1:  # 特徴量重要度（1回のみ）
            importances = model.feature_importance(importance_type="gain")
            feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(by="importance", ascending=False)
            plt.figure(figsize=(10, 8))
            sns.barplot(x="importance", y="feature", data=feature_importance_df.head(30), palette="viridis")
            plt.title("Top 30 Feature Importance (LightGBM)")
            plt.savefig("output/feature_importance_lgb.png")
            plt.close()
    metrics.append({"Model": f"LightGBM_i+{k}", "RMSE": np.mean(rmses), "MAE": np.mean(maes)})

# LSTM
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# LSTM用データ準備
timesteps = 10
X_lstm_3d = np.array([X_lstm.iloc[i-timesteps:i].values for i in range(timesteps, len(X_lstm))])
y_lstm = y.iloc[timesteps:]
X_lstm_3d = np.nan_to_num(X_lstm_3d, nan=0, posinf=1e6, neginf=-1e6)
lstm_metrics = []

for k in range(1, 6):
    rmses, maes = [], []
    for train_idx, test_idx in tscv.split(X_lstm_3d):
        X_train, X_test = X_lstm_3d[train_idx], X_lstm_3d[test_idx]
        y_train, y_test = y_lstm[f"Close_i+{k}"].iloc[train_idx], y_lstm[f"Close_i+{k}"].iloc[test_idx]
        model = create_lstm_model((timesteps, len(features_lstm)))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        model.save(f"model/LSTM_20250504/model_i+{k}.h5")
        y_pred = model.predict(X_test, verbose=0).flatten()
        y_pred = np.nan_to_num(y_pred, nan=0, posinf=1e6, neginf=-1e6)
        rmses.append(mean_squared_error(y_test, y_pred, squared=False))
        maes.append(mean_absolute_error(y_test, y_pred))
    lstm_metrics.append({"Model": f"LSTM_i+{k}", "RMSE": np.mean(rmses), "MAE": np.mean(maes)})

# メトリクス保存
metrics_df = pd.DataFrame(metrics + lstm_metrics)
metrics_df.to_csv("output/metrics.csv", index=False)

# 可視化
for k in range(1, 6):
    # LightGBM予測
    lgb_model = joblib.load(f"model/LightGBM_20250504/model_i+{k}.joblib")
    y_pred_lgb = lgb_model.predict(X)
    y_pred_lgb = np.nan_to_num(y_pred_lgb, nan=0, posinf=1e6, neginf=-1e6)
    # LSTM予測
    lstm_model = tf.keras.models.load_model(f"model/LSTM_20250504/model_i+{k}.h5")
    y_pred_lstm = lstm_model.predict(X_lstm_3d, verbose=0).flatten()
    y_pred_lstm = np.nan_to_num(y_pred_lstm, nan=0, posinf=1e6, neginf=-1e6)
    # 時系列プロット
    plt.figure(figsize=(12, 6))
    plt.plot(y.index[-100:], y[f"Close_i+{k}"][-100:], label="Actual", color="blue")
    plt.plot(y.index[-100:], y_pred_lgb[-100:], label="LightGBM", color="red", linestyle="--")
    plt.plot(y.index[-100:], y_pred_lstm[-100:], label="LSTM", color="green", linestyle="-.")
    plt.title(f"Prediction for Close_i+{k}")
    plt.xlabel("Date")
    plt.ylabel("Close Price (JPY)")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/timeseries_i+{k}.png")
    plt.close()
    # 誤差分布
    errors_lgb = y[f"Close_i+{k}"][-100:] - y_pred_lgb[-100:]
    errors_lstm = y[f"Close_i+{k}"][-100:] - y_pred_lstm[-100:]
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

# メトリクス比較
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="RMSE", data=metrics_df)
plt.title("RMSE Comparison")
plt.xticks(rotation=45)
plt.savefig("output/rmse_comparison.png")
plt.close()