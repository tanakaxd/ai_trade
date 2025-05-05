import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
import os
import yfinance as yf

# ディレクトリ作成
os.makedirs("data", exist_ok=True)

# データ取得（yfinance）
ticker = "7203.T"
data = yf.Ticker(ticker).history(period="60d", interval="5m")
print(data)
data = data[["Open", "High", "Low", "Close", "Volume"]]
print(data)

data.index = pd.to_datetime(data.index)
print(data)

data = data.sort_index()
print(data)


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

# 統計量
stat_features = {}
for window in [5, 10, 20]:
    stat_features[f"Close_mean_{window}"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}"] = data["Volume"].rolling(window=window).mean()
stat_df = pd.DataFrame(stat_features, index=data.index)

# 寄り/引け特徴量（先にラグ特徴量や統計量が必要なものを計算）
data = pd.concat([data, lag_df, stat_df], axis=1)  # ここで結合
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

open_close_features = {
    "Open_to_prev_close": data["Open"] - data["Close"].shift(1),
    "Open_to_sma5": data["Open"] - data["trend_sma_fast"],
    "Open_volatility": data["Open"].rolling(window=5).std(),
    "Close_to_sma5": data["Close"] - data["trend_sma_fast"],
    "Close_to_vwap": data["Close"] - data["volume_vwap"],
    "Close_volatility": data["Close"].rolling(window=5).std()
}
open_close_df = pd.DataFrame(open_close_features, index=data.index)

# 特徴量合成（先にラグ特徴量が必要）
synth_features = {
    "Close_lag_1_plus_Volume_lag_1": np.clip(data["Close_lag_1"] + data["Volume_lag_1"], -1e6, 1e6),
    "Close_lag_1_div_Volume_lag_1": np.clip(data["Close_lag_1"] / data["Volume_lag_1"].replace(0, np.nan), -1e6, 1e6),
    "RSI_minus_MACD": np.clip(data["momentum_rsi"] - data["trend_macd"], -1e6, 1e6),
    "Close_mean_5_plus_BBW": np.clip(data["Close_mean_5"] + data["volatility_bbw"], -1e6, 1e6),
    "Close_lag_5_div_SMA5": np.clip(data["Close_lag_5"] / data["trend_sma_fast"].replace(0, np.nan), -1e6, 1e6),
    "Close_lag_1_times_Volume_lag_1": np.clip(data["Close_lag_1"] * data["Volume_lag_1"], -1e6, 1e6)
}
synth_df = pd.DataFrame(synth_features, index=data.index)

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

# カテゴリカル変数
cat_features = {
    "hour": data.index.hour,
    "day_of_week": data.index.dayofweek,
    "is_opening": ((data.index.hour == 9) & (data.index.minute <= 30)).astype(int),
    "is_closing": ((data.index.hour == 14) & (data.index.minute >= 30)).astype(int)
}
cat_df = pd.DataFrame(cat_features, index=data.index)

# センチメント（ダミー）
sentiment = pd.Series(np.random.uniform(-1, 1, len(data)), index=data.index, name="sentiment")

# 全ての特徴量を一度に結合
data = pd.concat([data, open_close_df, synth_df, imbalance_df, cat_df, sentiment], axis=1)
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

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

# データ保存
data_dict = {
    "X": X,
    "y": y,
    "X_lstm": X_lstm,
    "y_scaled": y_scaled,
    "scalers_y": scalers_y,
    "features": features,
    "features_lstm": features_lstm
}
pd.to_pickle(data_dict, "data/processed_data.pkl")