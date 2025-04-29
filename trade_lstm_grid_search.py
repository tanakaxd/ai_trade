import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas_ta as ta
from itertools import product
from tqdm import tqdm
import uuid

# Load tickers from CSV
csv_data = pd.read_csv('daytrade_stocks.csv')
tickers = csv_data['ticker'].tolist()

# Parameter grid
sequence_lengths = [20, 50, 100]
lookahead_periods = [3, 5, 10]
scaling_factors = [50, 100, 200]
thresholds = [0.05, 0.1, 0.2]

# Results storage
results = []

# Grid search
for ticker, seq_len, look_ahead, scale_factor, thresh in tqdm(product(tickers, sequence_lengths, lookahead_periods, scaling_factors, thresholds), total=len(tickers) * len(sequence_lengths) * len(lookahead_periods) * len(scaling_factors) * len(thresholds)):
    try:
        # データ取得
        data = yf.download(ticker, interval="5m", start="2025-02-27", end="2025-04-27")
        if data.empty or len(data) < 100:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # テクニカル指標
        data['Returns'] = data['Close'].pct_change()
        data['Future_Return'] = (data['Close'].shift(-look_ahead) - data['Close']) / data['Close']
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['MACD'] = ta.macd(data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
        data['BB_Upper'] = ta.bbands(data['Close'], length=20, std=2)['BBU_20_2.0']
        data['BB_Lower'] = ta.bbands(data['Close'], length=20, std=2)['BBL_20_2.0']
        data['Price_Spread'] = (data['High'] - data['Low']) / data['Close']
        data['Sentiment_Proxy'] = data['Price_Spread'] * data['Volume']

        # データ準備
        features = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']
        data = data.dropna()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(len(scaled_data) - seq_len - look_ahead):
            X.append(scaled_data[i:i + seq_len])
            y.append(data['Future_Return'].iloc[i + seq_len])
        X, y = np.array(X), np.array(y)

        if len(X) < 50:
            continue

        # 訓練・テストデータ分割
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # LSTMモデル
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        # 予測
        predictions = model.predict(X_test, verbose=0)
        data['Predicted_Return'] = pd.Series(np.concatenate([np.zeros(train_size + seq_len + look_ahead), predictions.flatten()]), index=data.index)

        # ポジションサイジング
        max_position = 2.0
        data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * scale_factor, 0, max_position) * np.sign(data['Predicted_Return'])
        data['Position_Size'] = np.where(np.abs(data['Position_Size']) < thresh, 0, data['Position_Size'])

        # 戦略リターン
        data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
        transaction_cost = 0.001
        data['Strategy_Return'] = data['Strategy_Return'] - transaction_cost * data['Position_Size'].diff().abs()

        # 結果評価
        cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod().iloc[-1]
        sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252 * 66) if data['Strategy_Return'].std() != 0 else 0
        trade_count = int(data['Position_Size'].diff().abs().gt(0).sum())
        total_cost = transaction_cost * data['Position_Size'].diff().abs().sum()
        pred_return_std = data['Predicted_Return'].std()

        # Save results
        results.append({
            'ticker': ticker,
            'sequence_length': seq_len,
            'lookahead_period': look_ahead,
            'scaling_factor': scale_factor,
            'threshold': thresh,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': trade_count,
            'total_cost': total_cost,
            'pred_return_std': pred_return_std
        })

        # Save incrementally
        pd.DataFrame(results).to_csv('optimization_results.csv', index=False)

    except Exception as e:
        print(f"Error for {ticker}, seq_len={seq_len}, look_ahead={look_ahead}, scale_factor={scale_factor}, thresh={thresh}: {e}")
        continue

print("Optimization complete. Results saved to optimization_results.csv")