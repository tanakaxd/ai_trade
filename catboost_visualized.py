import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# データ取得
ticker = "7203.T"  # トヨタ
data = yf.download(ticker, start="2020-01-01", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=False)

# マルチインデックスの処理
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 特徴量生成
# 1. テクニカル指標（taライブラリ）
data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume"
)

# 2. ラグ特徴量（一括追加）
lag_features = {}
for lag in [1, 5, 10, 20, 50]:
    lag_features[f"Close_lag_{lag}"] = data["Close"].shift(lag)
    lag_features[f"Volume_lag_{lag}"] = data["Volume"].shift(lag)
    lag_features[f"Close_pct_change_{lag}"] = data["Close"].pct_change(periods=lag)
data = pd.concat([data, pd.DataFrame(lag_features, index=data.index)], axis=1)

# 3. 統計量（一括追加）
stat_features = {}
for window in [5, 20, 50]:
    stat_features[f"Close_mean_{window}d"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}d"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}d"] = data["Volume"].rolling(window=window).mean()
data = pd.concat([data, pd.DataFrame(stat_features, index=data.index)], axis=1)

# 4. カテゴリ特徴量
cat_features_dict = {
    "day_of_week": data.index.dayofweek,
    "month": data.index.month,
    "quarter": data.index.quarter
}
data = pd.concat([data, pd.DataFrame(cat_features_dict, index=data.index)], axis=1)

# 5. センチメント（ダミーデータ）
data["sentiment"] = np.random.uniform(-1, 1, len(data))

# 特徴量リスト確認
columns_to_drop = ["Close", "Open", "High", "Low", "Volume", "Adj Close"] if "Adj Close" in data.columns else ["Close", "Open", "High", "Low", "Volume"]
features = data.drop(columns=[col for col in columns_to_drop if col in data.columns]).columns
print(f"Total features: {len(features)}")

# データ準備
X = data[features].fillna(0)
y = data["Close"].shift(-1).fillna(method="ffill")  # 翌日終値
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# CatBoost学習
cat_features = ["day_of_week", "month", "quarter"]
model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, random_seed=42, verbose=100)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
model.fit(train_pool)

# 予測
test_pool = Pool(X_test, cat_features=cat_features)
y_pred = model.predict(test_pool)

# 評価指標
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# 可視化
# 1. 実測値 vs 予測値（時系列プロット）
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Close", color="blue", alpha=0.7)
plt.plot(y_test.index, y_pred, label="Predicted Close", color="red", linestyle="--")
plt.title(f"CatBoost Stock Price Prediction for {ticker}")
plt.xlabel("Date")
plt.ylabel("Close Price (JPY)")
plt.legend()
plt.grid()
plt.show()

# 2. 予測誤差の分布（ヒストグラム）
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True, FESTcolor="purple")
plt.title(f"Distribution of Prediction Errors (Mean: {errors.mean():.2f}, Std: {errors.std():.2f})")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 3. 特徴量重要度（トップ30）
importances = model.get_feature_importance(train_pool)
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)
top_30 = feature_importance_df.head(30)

plt.figure(figsize=(10, 8))
sns.barplot(x="importance", y="feature", data=top_30, palette="viridis")
plt.title("Top 30 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid()
plt.show()

# トップ30特徴量を表示
print("Top 30 features:", top_30["feature"].tolist())