import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
import pandas_ta as ta
import os
import joblib

# 設定パラメータ（前回と同じ）
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.2
MAX_POSITION = 2.0
TRANSACTION_COST = 0.001

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'
MODEL_NAME = 'trade_lstm_top_param_extended_analysis'  # 前回の実行ファイル名（拡張子なし）
TICKERS_CSV = 'daytrade_stocks.csv'

# モデルとscalerの保存パス
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME, 'lstm_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, MODEL_NAME, 'scaler.pkl')

# 特徴量リスト（前回と同じ）
FEATURES = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']

# 結果保存用のリスト
results = []

def load_tickers(csv_path):
    df = pd.read_csv(csv_path)
    return df['ticker'].tolist()

def process_data(data, scaler):
    # テクニカル指標の計算
    data['Returns'] = data['Close'].pct_change()
    data['Future_Return'] = (data['Close'].shift(-LOOKAHEAD_PERIOD) - data['Close']) / data['Close']
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['MACD'] = ta.macd(data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    data['BB_Upper'] = ta.bbands(data['Close'], length=20, std=2)['BBU_20_2.0']
    data['BB_Lower'] = ta.bbands(data['Close'], length=20, std=2)['BBL_20_2.0']
    data['Price_Spread'] = (data['High'] - data['Low']) / data['Close']
    data['Sentiment_Proxy'] = data['Price_Spread'] * data['Volume']

    # 欠損値除去
    data = data.dropna()

    # スケーリング（ロードしたscalerを使用）
    scaled_data = scaler.transform(data[FEATURES])

    return data, scaled_data

def create_sequences(scaled_data, data, seq_length, look_ahead):
    X = []
    indices = []
    for i in range(len(scaled_data) - seq_length - look_ahead + 1):
        X.append(scaled_data[i:(i + seq_length)])
        indices.append(i + seq_length + look_ahead - 1)
    return np.array(X), indices

def evaluate_ticker(ticker, model, scaler):
    try:
        print(f"\nティッカー {ticker} の処理を開始します...")
        
        # データ取得
        data = yf.download(ticker, interval="5m", start="2025-03-02", end="2025-04-29")
        if data.empty:
            print(f"{ticker}: データが取得できませんでした。")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # データ処理
        data, scaled_data = process_data(data, scaler)

        if len(data) < SEQUENCE_LENGTH + LOOKAHEAD_PERIOD:
            print(f"{ticker}: データが不足しています（{len(data)}行）。")
            return None

        # シーケンス作成
        X, seq_indices = create_sequences(scaled_data, data, SEQUENCE_LENGTH, LOOKAHEAD_PERIOD)

        if len(X) == 0:
            print(f"{ticker}: シーケンスデータが作成できませんでした。")
            return None

        # 予測
        print(f"{ticker}: 予測を生成中...")
        predictions = model.predict(X, verbose=0)

        # 予測結果をデータフレームに追加
        pred_df = pd.DataFrame({
            'Predicted_Return': predictions.flatten()
        }, index=data.index[seq_indices])
        data = data.join(pred_df, how='left')
        data['Predicted_Return'] = data['Predicted_Return'].fillna(0)

        # ポジションサイジング
        print(f"{ticker}: ポジションサイジングを計算中...")
        data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(data['Predicted_Return'])
        data['Position_Size'] = np.where(np.abs(data['Position_Size']) < THRESHOLD, 0, data['Position_Size'])

        # 戦略リターン
        print(f"{ticker}: 戦略リターンを計算中...")
        data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
        data['Strategy_Return'] = data['Strategy_Return'] - TRANSACTION_COST * data['Position_Size'].diff().abs()

        # 結果評価
        print(f"{ticker}: 結果を評価中...")
        cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod().iloc[-1]
        sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252 * 66) if data['Strategy_Return'].std() != 0 else 0
        trade_count = int(data['Position_Size'].diff().abs().gt(0).sum())
        total_cost = TRANSACTION_COST * data['Position_Size'].diff().abs().sum()
        pred_return_std = data['Predicted_Return'].std()

        # プロット
        plt.figure(figsize=(10, 6))
        (1 + data['Strategy_Return'].fillna(0)).cumprod().plot(label='Strategy Cumulative Return')
        plt.title(f'{ticker} Trading Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_strategy_performance.png'))
        plt.close()

        # 結果を保存
        result = {
            'ticker': ticker,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': trade_count,
            'total_cost': total_cost,
            'pred_return_std': pred_return_std
        }
        print(f"{ticker} のトレーディング結果:")
        print(f"累積リターン: {cumulative_return:.4f}")
        print(f"シャープレシオ: {sharpe_ratio:.4f}")
        print(f"取引回数: {trade_count}")
        print(f"総取引コスト: {total_cost:.4f}")
        print(f"予測リターンの標準偏差: {pred_return_std:.4f}")
        return result

    except Exception as e:
        print(f"{ticker}: エラーが発生しました: {e}")
        return None

def main():
    # モデルとscalerのロード
    print("モデルとscalerをロード中...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"モデルをロードしました: {MODEL_PATH}")
        print(f"Scalerをロードしました: {SCALER_PATH}")
    except Exception as e:
        print(f"モデルまたはscalerのロードに失敗しました: {e}")
        return

    # ティッカーリストのロード
    tickers = load_tickers(TICKERS_CSV)
    print(f"検証対象ティッカー: {tickers}")

    # 各ティッカーに対して評価
    for ticker in tickers:
        result = evaluate_ticker(ticker, model, scaler)
        if result:
            results.append(result)

    # 結果をCSVに保存
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'daytrade_stocks_results.csv'), index=False)
        print(f"\n結果を保存しました: {os.path.join(OUTPUT_DIR, 'daytrade_stocks_results.csv')}")
        print("\n最終結果:")
        print(results_df)
    else:
        print("結果がありませんでした。")

if __name__ == "__main__":
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()