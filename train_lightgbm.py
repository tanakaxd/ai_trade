import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import os
import joblib
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ディレクトリ作成
os.makedirs("output", exist_ok=True)
os.makedirs("model/LightGBM_20250504", exist_ok=True)

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X = data_dict["X"]
y_orig = data_dict["y_orig"]["Close_i+5"]
features = data_dict["features"]

# 追加特徴量
data = pd.DataFrame(X)
data["Close_current"] = data_dict["Close_current"]
data["High"] = data["Close_current"].rolling(window=2).max().shift(1)
data["Low"] = data["Close_current"].rolling(window=2).min().shift(1)
data["sma_200"] = SMAIndicator(data["Close_current"], window=200).sma_indicator()
data["atr"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close_current"], window=14).average_true_range()
bb = BollingerBands(data["Close_current"], window=20)
data["bb_width"] = bb.bollinger_wband()
X["sma_200"] = data["sma_200"]
X["atr"] = data["atr"]
X["bb_width"] = data["bb_width"]

# 重要度1未満の特徴量を除外
drop_features = ["Close_SMA5_diff", "Close_lag_1_plus_Volume_lag_1", "Close_mean_5", "Close_mean_20", 
                 "hour", "day_of_week", "is_opening", "is_closing", "Close_volatility",
                 "sentiment", "Volume_lag_1", "Close_lag_1_times_Volume_lag_1", "Close_lag_1_div_Volume_lag_1",
                 "Volume_lag_3", "Close_lag_3_div_Volume_lag_3", "High_Low_diff", "Close_lag_5_div_SMA5",
                 "Volume_lag_5", "Buy_Sell_Imbalance", "BBW_times_VWAP", "Volume_Imbalance", "Open_to_prev_close",
                 "Volume_lag_15", "Close_pct_change_15", "Volume_lag_30", "Open_to_sma5", "Close_Open_diff",
                 "Open_volatility", "Volume_lag_20", "Close_lag_10", "Volume_lag_10"]
selected_features = [f for f in X.columns if f not in drop_features]
X = X[selected_features]

# カテゴリカル変数（重要度0のためなし）
cat_columns = []

# ターゲットの対数変換とスケーリング
y_log = np.log1p(y_orig)
scaler_y = RobustScaler()
y = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
metrics = []

# LightGBM学習
rmses, maes, mapes = [], [], []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_test_orig = y_orig.iloc[test_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_columns)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.01,
        "n_estimators": 1500,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.train(params, train_data, valid_sets=[test_data], callbacks=[lgb.early_stopping(50, verbose=True)])
    y_pred = model.predict(X_test)
    
    # スケール戻しと指数変換
    y_pred_log = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.expm1(y_pred_log)
    y_pred_orig = np.clip(y_pred_orig, 300, 3500)  # 株価範囲を300〜3500に調整
    print(f"Fold {fold} - Final predicted y_orig range: min={y_pred_orig.min()}, max={y_pred_orig.max()}")
    
    # メトリクス計算
    rmse = mean_squared_error(y_test_orig, y_pred_orig, squared=False)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig.replace(0, np.nan))) * 100
    
    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    print(f"Fold {fold} - RMSE: {rmse}, MAE: {mae}, MAPE: {mape}%")

# モデル保存
joblib.dump(model, "model/LightGBM_20250504/model_i+5.joblib")

# 特徴量重要度を保存
importance = pd.DataFrame({
    "Feature": selected_features,
    "Importance": model.feature_importance(importance_type="gain")
})
importance = importance.sort_values(by="Importance", ascending=False)
importance.to_csv("output/feature_importance.csv", index=False)

# メトリクス保存
metrics.append({
    "Model": "LightGBM_i+5",
    "RMSE": np.mean(rmses),
    "MAE": np.mean(maes),
    "MAPE": np.mean(mapes)
})
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("output/lightgbm_metrics.csv", index=False)