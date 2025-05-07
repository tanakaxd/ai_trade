import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib

# ディレクトリ作成
os.makedirs("output", exist_ok=True)
os.makedirs("model/LightGBM_20250504", exist_ok=True)

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X = data_dict["X"]
y = data_dict["y"]["Close_i+5"]
y_orig = data_dict["y_orig"]["Close_i+5"]
scaler_y = data_dict["scaler_y"]
cat_columns = ["hour", "day_of_week", "is_opening", "is_closing"]
features = data_dict["features"]

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
metrics = []

# LightGBM学習
rmses, maes = [], []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_test_orig = y_orig.iloc[test_idx]
    
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
    y_pred = model.predict(X_test)
    
    # スケール戻し
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_orig = np.clip(y_pred_orig, 500, 5000)  # トヨタ株価の範囲を考慮
    
    # メトリクス計算
    rmse = mean_squared_error(y_test_orig, y_pred_orig, squared=False)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    
    rmses.append(rmse)
    maes.append(mae)

# モデル保存
joblib.dump(model, "model/LightGBM_20250504/model_i+5.joblib")

# メトリクス保存
metrics.append({
    "Model": "LightGBM_i+5",
    "RMSE": np.mean(rmses),
    "MAE": np.mean(maes)
})
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("output/lightgbm_metrics.csv", index=False)