import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt

# Fixed parameters
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.05
MAX_POSITION = 2.0
TRANSACTION_COST = 0.001
ROLLING_WINDOW = 252 * 66  # Approx 1 year of 5-min data (252 days * 66 intervals/day)

# File paths
MODEL_DIR = 'model'
COMBINED_DATA_PATH = os.path.join(MODEL_DIR, 'nikkei_combined_5min.csv')
RESULTS_PATH = os.path.join(MODEL_DIR, 'trading_results.csv')
PLOT_PATH = os.path.join(MODEL_DIR, 'strategy_performance.png')
FEATURES = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']

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

# 3. ローリングスケーリング
def rolling_scale_data(df, features, target_col, window):
    feature_data = df[features].values
    target_data = df[[target_col]].values
    scaled_features = np.zeros_like(feature_data)
    scaled_targets = np.zeros_like(target_data)
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    for i in range(len(df)):
        start = max(0, i - window)
        window_data = feature_data[start:i+1]
        window_target = target_data[start:i+1]
        
        if len(window_data) > 1:  # Ensure there's enough data to scale
            scaled_features[i] = feature_scaler.fit_transform(window_data)[-1]
            scaled_targets[i] = target_scaler.fit_transform(window_target)[-1]
        else:
            scaled_features[i] = feature_data[i]  # No scaling for the first window
            scaled_targets[i] = target_data[i]
    
    return scaled_features, scaled_targets, feature_scaler, target_scaler

# 4. シーケンスデータ作成
def create_sequences(data, seq_length, look_ahead):
    X, y = [], []
    indices = []
    for i in range(len(data) - seq_length - look_ahead + 1):
        X.append(data[i:(i + seq_length), :len(FEATURES)])
        y.append(data[i + seq_length + look_ahead - 1, len(FEATURES)])
        indices.append(i + seq_length + look_ahead - 1)
    return np.array(X), np.array(y), indices

# メイン処理
def main():
    print("データを読み込み中...")
    data = load_data(COMBINED_DATA_PATH)
    
    if data.empty or len(data) < 100:
        print("データが不足しています。")
        return
    
    print("特徴量を計算中...")
    data = engineer_features(data)
    
    # Debug: Check Future_Return
    print(f"Future_Return distribution:\n{data['Future_Return'].describe()}")
    
    # ローリングスケーリング
    print("ローリングスケーリングを実行中...")
    feature_data, target_data, feature_scaler, target_scaler = rolling_scale_data(
        data, FEATURES, 'Future_Return', ROLLING_WINDOW
    )
    scaled_data = np.hstack([feature_data, target_data])
    
    print("シーケンスデータを作成中...")
    X, y, seq_indices = create_sequences(scaled_data, SEQUENCE_LENGTH, LOOKAHEAD_PERIOD)
    
    if len(X) < 50:
        print("シーケンスデータが不足しています。")
        return
    
    # 訓練・テストデータ分割
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Debug: Check shapes and target distribution
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y distribution:\n{pd.Series(y).describe()}")
    
    # LSTMモデル
    print("モデルを構築中...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print("モデルを学習中...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'loss_plot.png'))
    plt.close()
    
    # 予測
    print("予測を生成中...")
    predictions = model.predict(X, verbose=0)
    
    # 予測結果をデータフレームに追加
    pred_df = pd.DataFrame({
        'Predicted_Return': predictions.flatten()
    }, index=data.index[seq_indices])
    
    data = data.join(pred_df, how='left')
    data['Predicted_Return'] = data['Predicted_Return'].fillna(0)
    
    # Debug: Check Predicted_Return
    print(f"Predicted_Return distribution:\n{data['Predicted_Return'].describe()}")
    
    # ポジションサイジング
    print("ポジションサイジングを計算中...")
    # 動的に閾値を調整（予測リターンの標準偏差に基づく）
    pred_std = data['Predicted_Return'].std()
    dynamic_threshold = max(THRESHOLD * pred_std / 0.0349, 0.01)  # 0.0349は限定期間の標準偏差
    print(f"Dynamic Threshold: {dynamic_threshold}")
    
    data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(data['Predicted_Return'])
    data['Position_Size'] = np.where(np.abs(data['Position_Size']) < dynamic_threshold, 0, data['Position_Size'])
    
    # 戦略リターン
    print("戦略リターンを計算中...")
    data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
    data['Strategy_Return'] = data['Strategy_Return'] - TRANSACTION_COST * data['Position_Size'].diff().abs()
    
    # Debug: Check Strategy_Return and trades
    print(f"Strategy_Return distribution:\n{data['Strategy_Return'].describe()}")
    trades = data[data['Position_Size'].diff().abs() > 0]
    print(f"Trades:\n{trades[['Position_Size', 'Predicted_Return', 'Future_Return', 'Strategy_Return']]}")
    
    # 結果評価
    print("結果を評価中...")
    cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod().iloc[-1]
    sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252 * 66) if data['Strategy_Return'].std() != 0 else 0
    trade_count = int(data['Position_Size'].diff().abs().gt(0).sum())
    total_cost = TRANSACTION_COST * data['Position_Size'].diff().abs().sum()
    pred_return_std = data['Predicted_Return'].std()
    
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
    (1 + data['Strategy_Return'].fillna(0)).cumprod().plot(label='Strategy Cumulative Return')
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