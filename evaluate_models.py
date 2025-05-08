import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import TimeSeriesSplit  # 追加

# パラメータ
RESULTS_PATH = os.path.join("output", "trading_results.csv")
TRADES_PATH = os.path.join("output", "trade_records.csv")
PLOT_PATH = os.path.join("output", "trading_performance.png")
SCALING_FACTOR = 200
THRESHOLD = 0.05
MAX_POSITION = 1.0
TRANSACTION_COST = 0.001

# トレーディング評価関数
def evaluate_trading(y_true, y_pred, close_prices, index, model_name, scaling_factor=200, threshold=0.05, max_position=1.0, transaction_cost=0.001):
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
    
    # メトリクス計算
    cumulative_return = (1 + strategy_return).cumprod()[-1] if len(strategy_return) > 0 else 0
    sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252) if np.std(strategy_return) != 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(1 + strategy_return) - 1 - np.minimum.accumulate(1 + strategy_return)) if len(strategy_return) > 0 else 0
    
    return cumulative_return, sharpe_ratio, max_drawdown, strategy_return, trades

# 年次評価関数
def evaluate_yearly(trades_df, model_name):
    trades_df["Year"] = trades_df["Timestamp"].dt.year
    yearly_results = []
    
    for year in trades_df["Year"].unique():
        year_data = trades_df[trades_df["Year"] == year]
        if len(year_data) == 0:
            continue
        strategy_return = year_data["Strategy_Return"].values
        cumulative_return = (1 + strategy_return).cumprod()[-1] if len(strategy_return) > 0 else 0
        annualized_return = ((1 + strategy_return).prod() ** (252 / len(strategy_return))) - 1 if len(strategy_return) > 0 else 0
        sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252) if np.std(strategy_return) != 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(1 + strategy_return) - 1 - np.minimum.accumulate(1 + strategy_return)) if len(strategy_return) > 0 else 0
        
        yearly_results.append({
            "Model": model_name,
            "Year": year,
            "Cumulative_Return": cumulative_return,
            "Annualized_Return": annualized_return,
            "Sharpe_Ratio": sharpe_ratio,
            "Max_Drawdown": max_drawdown
        })
    
    return pd.DataFrame(yearly_results)

# メイン処理
def main():
    # ディレクトリ作成
    os.makedirs("output", exist_ok=True)
    
    # 予測データ読み込み
    predictions_df = pd.read_csv("output/predictions.csv", parse_dates=["Timestamp"])
    predictions_df = predictions_df.sort_values("Timestamp")
    
    # 訓練データ区間の除外（train_lightgbm.pyのtrain_idxを模擬的に取得）
    train_data = pd.read_pickle("data/processed_data_1d.pkl")
    tscv = TimeSeriesSplit(n_splits=5)
    train_indices = []
    for train_idx, _ in tscv.split(train_data["X"]):
        train_indices.extend(train_data["X"].index[train_idx])
    train_indices = pd.to_datetime(train_indices).unique()
    test_predictions_df = predictions_df[~predictions_df["Timestamp"].isin(train_indices)]
    
    # データ準備
    y_true = test_predictions_df["Actual_Close_i+5"].values
    y_pred = test_predictions_df["Predicted_Close_i+5"].values
    close_prices = test_predictions_df["Close_current"].values
    index = test_predictions_df["Timestamp"]
    
    # トレーディング評価
    results = []
    all_trades = []
    strategy_returns = {}
    
    # LightGBM評価
    cum_return, sharpe_ratio, max_drawdown, strategy_return, trades = evaluate_trading(
        y_true, y_pred, close_prices, index, "LightGBM",
        SCALING_FACTOR, THRESHOLD, MAX_POSITION, TRANSACTION_COST
    )
    results.append({
        "Model": "LightGBM",
        "Cumulative_Return": cum_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2))
    })
    strategy_returns["LightGBM"] = pd.Series(strategy_return, index=index)
    all_trades.append(trades)
    
    # 年次評価
    yearly_results_df = evaluate_yearly(trades, "LightGBM")
    
    # トレード記録保存
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(TRADES_PATH, index=False)
    
    # 結果保存
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(RESULTS_PATH, index=False)
    yearly_results_df.to_csv(os.path.join("output", "yearly_trading_results.csv"), index=False)
    
    # 累積リターン比較プロット
    plt.figure(figsize=(10, 6))
    for model, returns in strategy_returns.items():
        (1 + returns).cumprod().plot(label=f"{model} Cumulative Return")
    plt.title("Trading Strategy Performance (Toyota: 7203.T)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_PATH)
    plt.close()
    
    # 年次結果プロット
    plt.figure(figsize=(10, 6))
    yearly_results_df.plot(x="Year", y=["Cumulative_Return", "Annualized_Return", "Max_Drawdown"], kind="bar", title="Yearly Trading Performance")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend(["Cumulative Return", "Annualized Return", "Max Drawdown"])
    plt.savefig(os.path.join("output", "yearly_performance.png"))
    plt.close()

if __name__ == "__main__":
    main()