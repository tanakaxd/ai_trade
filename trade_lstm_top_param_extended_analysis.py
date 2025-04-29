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
import joblib  # scalerの保存用

# Parameter grid
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.2
MAX_POSITION = 2.0
TRANSACTION_COST = 0.001
ticker = "7974.T"

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'
FILE_NAME = os.path.basename(__file__).replace('.py', '')  # 拡張子を除去

# Results storage
results = []

# メイン処理を関数として定義
def main():
    try:
        # データ取得
        data = yf.download(ticker, interval="5m", start="2025-03-02", end="2025-04-29")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # テクニカル指標
        data['Returns'] = data['Close'].pct_change()
        data['Future_Return'] = (data['Close'].shift(-LOOKAHEAD_PERIOD) - data['Close']) / data['Close']
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
        for i in range(len(scaled_data) - SEQUENCE_LENGTH - LOOKAHEAD_PERIOD):
            X.append(scaled_data[i:i + SEQUENCE_LENGTH])
            y.append(data['Future_Return'].iloc[i + SEQUENCE_LENGTH])
        X, y = np.array(X), np.array(y)

        # 訓練・テストデータ分割
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # LSTMモデル
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train, 
            epochs=30, 
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Plot training history
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
        plt.close()

        # モデル保存ディレクトリを作成
        save_dir = os.path.join(MODEL_DIR, FILE_NAME)
        os.makedirs(save_dir, exist_ok=True)

        # モデル保存
        model_save_path = os.path.join(save_dir, 'lstm_model.h5')
        model.save(model_save_path)
        print(f"モデルを保存しました: {model_save_path}")

        # scaler保存
        scaler_save_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_save_path)
        print(f"Scalerを保存しました: {scaler_save_path}")

        # 予測
        predictions = model.predict(X_test, verbose=0)
        data['Predicted_Return'] = pd.Series(np.concatenate([np.zeros(train_size + SEQUENCE_LENGTH + LOOKAHEAD_PERIOD), predictions.flatten()]), index=data.index)

        # ポジションサイジング
        data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(data['Predicted_Return'])
        data['Position_Size'] = np.where(np.abs(data['Position_Size']) < THRESHOLD, 0, data['Position_Size'])

        # 戦略リターン
        data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
        data['Strategy_Return'] = data['Strategy_Return'] - TRANSACTION_COST * data['Position_Size'].diff().abs()

        # 結果評価
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
        pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR,'optimization_results.csv'), index=False)

        # プロット
        plt.figure(figsize=(10, 6))
        (1 + data['Strategy_Return'].fillna(0)).cumprod().plot(label='Strategy Cumulative Return')
        plt.title(f'{FILE_NAME} Trading Strategy Performance')
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

if __name__ == "__main__":
    main()
    print("Optimization complete. Results saved to optimization_results.csv")