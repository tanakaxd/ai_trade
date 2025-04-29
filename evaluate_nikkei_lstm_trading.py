import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from tensorflow.keras.models import load_model # type: ignore
import pickle
import matplotlib.pyplot as plt

# パラメータ
MODEL_DIR = 'model'
COMBINED_DATA_PATH = os.path.join(MODEL_DIR, 'nikkei_combined_5min.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'nikkei_lstm_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'trading_results.csv')
PLOT_PATH = os.path.join(MODEL_DIR, 'strategy_performance.png')
SEQUENCE_LENGTH = 50
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.05
FEATURES = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']
MAX_POSITION = 1.0
TRANSACTION_COST = 0.001

# 1. データ読み込み
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
        df = df.drop(columns=df.columns[0])
    df = df.set_index('datetime')
    if df.index.duplicated().any():
        print(f"Found {df.index.duplicated().sum()} duplicate timestamps. Dropping duplicates.")
        df = df[~df.index.duplicated(keep='first')]
    df = df.dropna()
    return df

# 2. 特徴量エンジニアリング
def engineer_features(df):
    df = df.copy()
    
    df['Close'] = df['close']
    df['Returns'] = df['close'].pct_change()
    df['Future_Return'] = (df['close'].shift(-LOOKAHEAD_PERIOD) - df['close']) / df['close']
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    df['Price_Spread'] = (df['high'] - df['low']) / df['close']
    df['Sentiment_Proxy'] = df['Price_Spread'] * df['volume']
    
    df[FEATURES + ['Future_Return']] = df[FEATURES + ['Future_Return']].replace([np.inf, -np.inf], np.nan)
    df[FEATURES + ['Future_Return']] = df[FEATURES + ['Future_Return']].interpolate(method='linear').fillna(method='bfill')
    df = df.dropna(subset=FEATURES + ['Future_Return'])
    
    return df

# 3. シーケンスデータ作成
def create_sequences(data, seq_length):
    X = []
    indices = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:(i + seq_length), :len(FEATURES)])
        indices.append(i + seq_length - 1)
    return np.array(X), indices

# メイン処理
def main():
    print("データを読み込み中...")
    df = load_data(COMBINED_DATA_PATH)
    
    print("特徴量を計算中...")
    df = engineer_features(df)
    
    # Debug: Check Future_Return and Predicted_Return
    print(f"Future_Return distribution:\n{df['Future_Return'].describe()}")
    
    print("スケーラーとモデルを読み込み中...")
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
        feature_scaler = scalers['feature']
        target_scaler = scalers['target']
    
    # モデルを読み込む
    model = load_model(MODEL_PATH)
    
    # 特徴量をスケーリング
    feature_data = df[FEATURES].values
    target_data = df[['Future_Return']].values
    feature_data = feature_scaler.transform(feature_data)
    target_data = target_scaler.transform(target_data)
    feature_data = np.hstack([feature_data, target_data])
    
    print("シーケンスデータを作成中...")
    X, seq_indices = create_sequences(feature_data, SEQUENCE_LENGTH)
    
    # 予測
    print("予測を生成中...")
    predictions = model.predict(X, verbose=0)
    
    # 予測値を元のスケールに戻す
    predictions = target_scaler.inverse_transform(predictions)
    
    # 予測結果をデータフレームに追加
    pred_df = pd.DataFrame({
        'Predicted_Return': predictions.flatten()
    }, index=df.index[seq_indices])
    
    df = df.join(pred_df, how='left')
    df['Predicted_Return'] = df['Predicted_Return'].fillna(0)
    
    # Debug: Check Predicted_Return
    print(f"Predicted_Return distribution:\n{df['Predicted_Return'].describe()}")
    
    # ポジションサイジング
    df['Position_Size'] = np.clip(np.abs(df['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(df['Predicted_Return'])
    df['Position_Size'] = np.where(np.abs(df['Position_Size']) < THRESHOLD, 0, df['Position_Size'])
    
    # 戦略リターン
    df['Strategy_Return'] = df['Position_Size'] * df['Future_Return']
    df['Strategy_Return'] = df['Strategy_Return'] - TRANSACTION_COST * df['Position_Size'].diff().abs()
    
    # Debug: Check Strategy_Return and trades
    print(f"Strategy_Return distribution:\n{df['Strategy_Return'].describe()}")
    trades = df[df['Position_Size'].diff().abs() > 0]
    print(f"Trades:\n{trades[['Position_Size', 'Predicted_Return', 'Future_Return', 'Strategy_Return']]}")
    
    # 結果評価
    cumulative_return = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1]
    sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252 * 66) if df['Strategy_Return'].std() != 0 else 0
    trade_count = int(df['Position_Size'].diff().abs().gt(0).sum())
    total_cost = TRANSACTION_COST * df['Position_Size'].diff().abs().sum()
    pred_return_std = df['Predicted_Return'].std()
    
    # 結果を保存
    results = [{
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': trade_count,
        'total_cost': total_cost,
        'pred_return_std': pred_return_std
    }]
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    
    # プロット
    plt.figure(figsize=(10, 6))
    (1 + df['Strategy_Return'].fillna(0)).cumprod().plot(label='Strategy Cumulative Return')
    plt.title('Nikkei LSTM Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()
    
    print("トレーディング結果:")
    print(f"累積リターン: {cumulative_return:.4f}")
    print(f"シャープレシオ: {sharpe_ratio:.4f}")
    print(f"取引回数: {trade_count}")
    print(f"総取引コスト: {total_cost:.4f}")
    print(f"予測リターンの標準偏差: {pred_return_std:.4f}")
    print(f"結果を保存: {RESULTS_PATH}")
    print(f"パフォーマンスプロット: {PLOT_PATH}")

if __name__ == '__main__':
    main()