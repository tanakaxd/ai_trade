import pandas as pd
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# パラメータ
MODEL_DIR = "model"
GRU_MODEL_PATH = os.path.join(MODEL_DIR, "GRU_20250504/model_fold_5.h5")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "LightGBM_20250504/model_i+5.joblib")
RESULTS_PATH = os.path.join("output", "trading_results.csv")
TRADES_PATH = os.path.join("output", "trade_records.csv")
PLOT_PATH = os.path.join("output", "trading_performance.png")
SCALING_FACTOR = 200
THRESHOLD = 0.05
MAX_POSITION = 1.0
TRANSACTION_COST = 0.001
TIMESTEPS = 50

# トレーディング評価関数
def evaluate_trading(y_true, y_pred, close_prices, index, model_name, scaling_factor=200, threshold=0.05, max_position=1.0, transaction_cost=0.001):
    # インデックスを揃える
    valid_idx = y_true.index.intersection(close_prices.index).intersection(index)
    y_true = y_true.loc[valid_idx].values
    y_pred = y_pred[:len(valid_idx)]  # y_predはnumpy配列なのでスライス
    close_prices = close_prices.loc[valid_idx].values
    index = valid_idx
    
    # 形状確認
    if len(y_true) != len(y_pred) or len(y_pred) != len(close_prices):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, close_prices={len(close_prices)}")
    
    predicted_return = (y_pred - close_prices) / close_prices
    position_size = np.clip(np.abs(predicted_return) * scaling_factor, 0, max_position) * np.sign(predicted_return)
    position_size = np.where(np.abs(position_size) < threshold, 0, position_size)
    future_return = (y_true - close_prices) / close_prices
    strategy_return = position_size * future_return
    transaction_costs = transaction_cost * np.abs(np.diff(position_size, prepend=0))
    strategy_return = strategy_return - transaction_costs
    
    # トレード記録
    trades = pd.DataFrame({
        "Timestamp": index,
        "Model": model_name,
        "Position_Size": position_size,
        "Predicted_Return": predicted_return,
        "Future_Return": future_return,
        "Strategy_Return": strategy_return,
        "Transaction_Cost": transaction_costs
    })
    
    # メトリクス
    cumulative_return = (1 + strategy_return).cumprod()[-1] if len(strategy_return) > 0 else 0
    sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252) if np.std(strategy_return) != 0 else 0
    trade_count = np.sum(np.abs(np.diff(position_size, prepend=0)) > 0)
    total_cost = transaction_cost * np.abs(np.diff(position_size, prepend=0)).sum()
    
    return cumulative_return, sharpe_ratio, trade_count, total_cost, strategy_return, trades

# シーケンスデータ作成（GRU用）
def create_sequences(data, seq_length):
    X = []
    indices = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:(i + seq_length)])
        indices.append(i + seq_length - 1)
    return np.array(X), indices

# メイン処理
def main():
    # ディレクトリ作成
    os.makedirs("output", exist_ok=True)
    
    # データ読み込み
    data_dict = pd.read_pickle("data/processed_data.pkl")
    X = data_dict["X"]
    X_gru = data_dict["X_gru"]
    y = data_dict["y_orig"]["Close_i+5"]
    close_prices = data_dict["Close_current"]
    scaler_y = data_dict["scaler_y"]
    features_gru = data_dict["features_gru"]
    
    # GRUデータ準備
    X_gru_3d, seq_indices = create_sequences(X_gru, TIMESTEPS)
    valid_idx = y.index[TIMESTEPS:]  # GRUのインデックスを調整
    y_gru = y.loc[valid_idx]
    close_gru = close_prices.loc[valid_idx]
    index_gru = valid_idx
    
    # LightGBMデータ準備
    valid_idx_lgbm = y.index.intersection(X.index).intersection(close_prices.index)
    y_lgbm = y.loc[valid_idx_lgbm]
    close_lgbm = close_prices.loc[valid_idx_lgbm]
    X_lgbm = X.loc[valid_idx_lgbm]
    index_lgbm = valid_idx_lgbm
    
    # モデル読み込み
    gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    
    # 予測
    y_pred_gru = gru_model.predict(X_gru_3d, verbose=0).flatten()
    y_pred_gru_orig = scaler_y.inverse_transform(y_pred_gru.reshape(-1, 1)).flatten()
    y_pred_gru_orig = np.clip(y_pred_gru_orig, 500, 5000)  # トヨタ株価の範囲
    
    y_pred_lgbm = lgbm_model.predict(X_lgbm)
    y_pred_lgbm_orig = scaler_y.inverse_transform(y_pred_lgbm.reshape(-1, 1)).flatten()
    y_pred_lgbm_orig = np.clip(y_pred_lgbm_orig, 500, 5000)
    
    # トレーディング評価
    results = []
    all_trades = []
    strategy_returns = {}
    
    # GRU評価
    cum_return, sharpe, trade_count, total_cost, strategy_return, trades = evaluate_trading(
        y_gru, y_pred_gru_orig, close_gru, index_gru, "GRU",
        SCALING_FACTOR, THRESHOLD, MAX_POSITION, TRANSACTION_COST
    )
    results.append({
        "Model": "GRU",
        "Cumulative_Return": cum_return,
        "Sharpe_Ratio": sharpe,
        "Trade_Count": trade_count,
        "Total_Cost": total_cost,
        "RMSE": np.sqrt(np.mean((y_gru - y_pred_gru_orig[:len(y_gru)]) ** 2)) if len(y_gru) > 0 else 0
    })
    strategy_returns["GRU"] = pd.Series(strategy_return, index=index_gru)
    all_trades.append(trades)
    
    # LightGBM評価
    cum_return, sharpe, trade_count, total_cost, strategy_return, trades = evaluate_trading(
        y_lgbm, y_pred_lgbm_orig, close_lgbm, index_lgbm, "LightGBM",
        SCALING_FACTOR, THRESHOLD, MAX_POSITION, TRANSACTION_COST
    )
    results.append({
        "Model": "LightGBM",
        "Cumulative_Return": cum_return,
        "Sharpe_Ratio": sharpe,
        "Trade_Count": trade_count,
        "Total_Cost": total_cost,
        "RMSE": np.sqrt(np.mean((y_lgbm - y_pred_lgbm_orig[:len(y_lgbm)]) ** 2)) if len(y_lgbm) > 0 else 0
    })
    strategy_returns["LightGBM"] = pd.Series(strategy_return, index=index_lgbm)
    all_trades.append(trades)
    
    # トレード記録保存
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(TRADES_PATH, index=False)
    
    # 結果保存
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(RESULTS_PATH, index=False)
    
    # プロット（時系列）
    plt.figure(figsize=(12, 6))
    plt.plot(index_lgbm[-100:], y_lgbm[-100:], label="Actual", color="blue")
    plt.plot(index_lgbm[-100:], y_pred_lgbm_orig[-100:], label="LightGBM", color="red", linestyle="--")
    plt.plot(index_gru[-100:], y_pred_gru_orig[-100:], label="GRU", color="green", linestyle="-.")
    plt.title("Prediction for Close_i+5 (Toyota: 7203.T)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (JPY)")
    plt.legend()
    plt.grid()
    plt.savefig("output/timeseries_i+5.png")
    plt.close()
    
    # エラー分布
    errors_lgbm = y_lgbm[-100:] - y_pred_lgbm_orig[-100:]
    errors_gru = y_gru[-100:] - y_pred_gru_orig[-100:]
    plt.figure(figsize=(8, 6))
    sns.histplot(errors_lgbm, bins=50, kde=True, color="purple", label="LightGBM")
    sns.histplot(errors_gru, bins=50, kde=True, color="green", label="GRU", alpha=0.5)
    plt.title("Error Distribution for Close_i+5")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.savefig("output/error_dist_i+5.png")
    plt.close()
    
    # 累積リターン比較プロット
    plt.figure(figsize=(10, 6))
    for model, returns in strategy_returns.items():
        (1 + returns).cumprod().plot(label=f"{model} Cumulative Return")
    plt.title("Trading Strategy Performance Comparison (Toyota: 7203.T)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_PATH)
    plt.close()
    
    # RMSE比較プロット
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="RMSE", data=metrics_df)
    plt.title("RMSE Comparison")
    plt.xticks(rotation=45)
    plt.savefig("output/rmse_comparison.png")
    plt.close()
    
    # 累積リターン比較プロット（棒グラフ）
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Cumulative_Return", data=metrics_df)
    plt.title("Cumulative Return Comparison")
    plt.xticks(rotation=45)
    plt.savefig("output/cumulative_return_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()