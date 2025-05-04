import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import numpy as np

# データ取得
ticker = "7203.T"
data = yf.download(ticker, start="2020-01-01", end="2025-05-04")

# マルチインデックスの場合の処理
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(data.columns)
print(data)


# 特徴量生成
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
for lag in [1, 5, 20]:
    data[f"Close_lag_{lag}"] = data["Close"].shift(lag)
data["Close_pct_change_1d"] = data["Close"].pct_change()
data["sentiment"] = np.random.uniform(-1, 1, len(data))  # ダミー
data["day_of_week"] = data.index.dayofweek

print(data.columns)
print(data)

data.to_csv("output/toyoda_add_all_ta_features.csv")


# データ準備
X = data.drop(columns=["Close"]).fillna(0)
y = data["Close"].shift(-1).fillna(method="ffill")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# CatBoost学習
cat_features = ["day_of_week"]
model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, random_seed=42)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
model.fit(train_pool)

# 重要度取得
importances = model.get_feature_importance(train_pool)
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# トップ30選択
top_30_features = feature_importance_df["feature"].head(30).tolist()
print("Top 30 features:", top_30_features)

# トップ30で再学習
X_train_top30 = X_train[top_30_features]
train_pool_top30 = Pool(X_train_top30, y_train, cat_features=[f for f in cat_features if f in top_30_features])
model_top30 = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1)
model_top30.fit(train_pool_top30)