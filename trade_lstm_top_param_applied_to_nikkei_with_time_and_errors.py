import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import pandas_ta as ta
from itertools import product
from tqdm import tqdm
import os
import joblib

# Parameter grid
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.2
MAX_POSITION = 2.0
TRANSACTION_COST = 0.001
ticker = "Nikkei225"  # Placeholder for ticker, as we are using CSV data

# Constants
MODEL_DIR = 'model'
OUTPUT_DIR = 'output'
MASTER_DATA_DIR = 'master_data'
INPUT_CSV = os.path.join(MASTER_DATA_DIR, 'nikkei_combined_5min_cleaned.csv')
FILE_NAME_RUNNING = os.path.basename(__file__).replace('.py', '')  # 拡張子を除去

start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-07-01')

# Results storage
results = []

try:
    # データ取得
    print("データを読み込み中...")
    data = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]
    data = data.set_index('Datetime')
    data = data.dropna()

    # 時間帯特徴の追加
    print("時間帯特徴を追加中...")
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    # 周期性を考慮して正弦・余弦変換
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_minute'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['cos_minute'] = np.cos(2 * np.pi * data['minute'] / 60)

    # テクニカル指標
    print("テクニカル指標を計算中...")
    data['Returns'] = data['Close'].pct_change()
    data['Future_Return'] = (data['Close'].shift(-LOOKAHEAD_PERIOD) - data['Close']) / data['Close']
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['MACD'] = ta.macd(data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    data['BB_Upper'] = ta.bbands(data['Close'], length=20, std=2)['BBU_20_2.0']
    data['BB_Lower'] = ta.bbands(data['Close'], length=20, std=2)['BBL_20_2.0']
    data['Price_Spread'] = (data['High'] - data['Low']) / data['Close']
    data['Sentiment_Proxy'] = data['Price_Spread'] * data['Volume']

    # データ準備
    features = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy', 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']
    data = data.dropna()
    # 異常値のクリーニング
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    data = data.dropna()

    print("データをスケーリング中...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    print("シーケンスを作成中...")
    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH - LOOKAHEAD_PERIOD):
        X.append(scaled_data[i:i + SEQUENCE_LENGTH])
        y.append(data['Future_Return'].iloc[i + SEQUENCE_LENGTH])
    X, y = np.array(X), np.array(y)

    # 訓練・テストデータ分割
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # LSTMモデル
    print("モデルを構築中...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("モデルを学習中...")
    history = model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Plot training history
    print("学習曲線をプロット中...")
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    plt.close()

    # モデルとscalerの保存
    print("モデルとscalerを保存中...")
    save_dir = os.path.join(MODEL_DIR, FILE_NAME_RUNNING)
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print(f"モデルを保存しました: {os.path.join(save_dir, 'lstm_model.h5')}")
    print(f"Scalerを保存しました: {os.path.join(save_dir, 'scaler.pkl')}")

    # 予測。予測自体は全データに対して行う
    print("予測を生成中...")
    predictions = model.predict(X, verbose=0)
    
    # デバッグ情報の出力
    print("\n=== デバッグ情報 ===")
    print(f"全データの長さ: {len(data)}")
    print(f"予測データの長さ: {len(predictions)}")
    print(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"最初の予測可能日: {data.index[SEQUENCE_LENGTH]}")
    print(f"最後の予測可能日: {data.index[SEQUENCE_LENGTH + len(predictions) - 1]}")
    print(f"データの最初の日付: {data.index[0]}")
    print(f"データの最後の日付: {data.index[-1]}")
    
    # 予測値をデータフレームに追加。SEQUENCE_LENGTH分のデータは予測できないため、その分をNaNで埋める
    data['Predicted_Return'] = np.nan
    data.loc[data.index[SEQUENCE_LENGTH:len(predictions)+SEQUENCE_LENGTH], 'Predicted_Return'] = predictions.flatten()
    
    # 予測値の分布を確認
    print("\n予測値の分布:")
    print(data['Predicted_Return'].describe())
    print(f"NaNの数: {data['Predicted_Return'].isna().sum()}")
    
    # ポジションサイジング
    print("\nポジションサイジングを計算中...")
    data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(data['Predicted_Return'])
    data['Position_Size'] = np.where(np.abs(data['Position_Size']) < THRESHOLD, 0, data['Position_Size'])
    # NaNを0に置き換える
    data['Position_Size'] = data['Position_Size'].fillna(0)
    
    # ポジションサイズの分布を確認
    print("\nポジションサイズの分布:")
    print(data['Position_Size'].describe())
    print(f"ポジションサイズが0でないデータ数: {(data['Position_Size'] != 0).sum()}")
    
    # 取引の発生を確認
    position_changes = data['Position_Size'].diff().abs() > 0
    print(f"\n取引の発生回数: {position_changes.sum()}")
    print(f"最初の取引日: {data[position_changes].index[0]}")
    print(f"最後の取引日: {data[position_changes].index[-1]}")
    print("==================\n")

    # 予測リターンと実際のリターンの乖離を計算
    print("予測と実際のリターンの乖離を計算中...")
    data['Prediction_Error'] = data['Predicted_Return'] - data['Future_Return']
    error_stats = data['Prediction_Error'].describe()
    print("Prediction Error distribution:")
    print(error_stats)
    # 乖離データをCSVに保存
    data[['Predicted_Return', 'Future_Return', 'Prediction_Error']].to_csv(
        os.path.join(OUTPUT_DIR, 'prediction_errors.csv')
    )
    print(f"乖離データを保存しました: {os.path.join(OUTPUT_DIR, 'prediction_errors.csv')}")

    # 戦略リターン
    print("戦略リターンを計算中...")
    data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
    data['Strategy_Return'] = data['Strategy_Return'] - TRANSACTION_COST * data['Position_Size'].diff().abs()

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

    # Save results
    results.append({
        'ticker': ticker,
        'sequence_length': SEQUENCE_LENGTH,
        'lookahead_period': LOOKAHEAD_PERIOD,
        'scaling_factor': SCALING_FACTOR,
        'threshold': THRESHOLD,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': trade_count,
        'total_cost': total_cost,
        'pred_return_std': pred_return_std
    })

    # Save incrementally
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'optimization_results.csv'), index=False)

    # プロット
    print("パフォーマンスをプロット中...")
    plt.figure(figsize=(10, 6))
    (1 + data['Strategy_Return'].fillna(0)).cumprod().plot(label='Strategy Cumulative Return')
    plt.title(f'{FILE_NAME_RUNNING} Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'strategy_performance.png'))
    plt.close()
    
    print("トレーディング結果:")
    print(f"累積リターン: {cumulative_return:.4f}")
    print(f"シャープレシオ: {sharpe_ratio:.4f}")
    print(f"取引回数: {trade_count}")
    print(f"総取引コスト: {total_cost:.4f}")
    print(f"予測リターンの標準偏差: {pred_return_std:.4f}")

except Exception as e:
    print(f"Error for {ticker}, SEQUENCE_LENGTH={SEQUENCE_LENGTH}, LOOKAHEAD_PERIOD={LOOKAHEAD_PERIOD}, SCALING_FACTOR={SCALING_FACTOR}, thresh={THRESHOLD}: {e}")

print("Optimization complete. Results saved to optimization_results.csv")