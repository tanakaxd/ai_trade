import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import joblib
import os

# ディレクトリ作成
os.makedirs("output", exist_ok=True)
os.makedirs("model/LightGBM_20250508", exist_ok=True)

# データ読み込み
data_dict = pd.read_pickle("data/processed_data_1d.pkl")
X = data_dict["X"]
y_orig = data_dict["y_orig"]["Close_i+5"]
features = data_dict["features"]
selected_features = features  # 全特徴量を使用

# カテゴリカル変数
cat_columns = data_dict.get("cat_columns", [])
cat_columns = [col for col in cat_columns if col in X.columns]  # データに存在する列のみ使用
print(f"Categorical columns used: {cat_columns}")

# ターゲットの対数変換とスケーリング
y_log = np.log1p(y_orig)
scaler_y = RobustScaler()
y = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()

# 全体の時系列プロット
plt.figure(figsize=(12, 6))
plt.plot(X.index, data_dict["Close_current"], label="Close_current")
plt.plot(X.index, y_orig, label="Close_i+5", alpha=0.5)
plt.title("Time Series of Close_current and Close_i+5")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("output/time_series_plot.png")
plt.close()

# 異常値チェック（ボックスプロット）
plt.figure(figsize=(8, 4))
plt.boxplot([data_dict["Close_current"], y_orig], labels=["Close_current", "Close_i+5"])
plt.title("Box Plot of Close_current and Close_i+5")
plt.savefig("output/box_plot.png")
plt.close()

# k-Fold Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
metrics = []
rmses, maes, mapes = [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_test_orig = y_orig.iloc[test_idx]
    
    # 日付範囲の表示
    train_dates = X.index[train_idx]
    test_dates = X.index[test_idx]
    print(f"Fold {fold} - Train date range: {train_dates[0]} to {train_dates[-1]}")
    print(f"Fold {fold} - Test date range: {test_dates[0]} to {test_dates[-1]}")
    
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
    
    # 予測プロット
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_orig, label="Actual Close_i+5", color="blue")
    plt.plot(test_dates, y_pred_orig, label="Predicted Close_i+5", color="orange", linestyle="--")
    plt.title(f"Fold {fold}: Actual vs Predicted Close_i+5")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/fold{fold}_predictions_vs_actual.png")
    plt.close()
    
    # 誤差プロット
    errors = y_test_orig - y_pred_orig
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, errors, label="Prediction Error (Actual - Predicted)", color="red")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.title(f"Fold {fold}: Prediction Errors")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(f"output/fold{fold}_prediction_errors.png")
    plt.close()
    
    # フォールド5のテストデータプロット
    if fold == 5:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, data_dict["Close_current"].iloc[test_idx], label="Close_current (Fold 5 Test)")
        plt.plot(test_dates, y_test_orig, label="Close_i+5 (Fold 5 Test)", alpha=0.5)
        plt.title("Fold 5 Test Data: Close_current and Close_i+5")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.savefig("output/fold5_test_plot.png")
        plt.close()

# モデル保存
joblib.dump(model, "model/LightGBM_20250508/model_i+5.joblib")

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