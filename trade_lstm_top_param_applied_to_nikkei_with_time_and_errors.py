import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import pandas_ta as ta
from itertools import product
from tqdm import tqdm
import os
import joblib

# 設定
TRAIN_MODEL = False  # True: モデルを学習, False: 保存済みモデルをロード
MODEL_VERSION = 'v1'  # モデルのバージョン管理用

# Parameter grid
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.2
MAX_POSITION = 2.0
TRANSACTION_COST = 0.001
ticker = "Nikkei225"

# Constants
MODEL_DIR = 'model'
OUTPUT_DIR = 'output'
MASTER_DATA_DIR = 'master_data'
INPUT_CSV = os.path.join(MASTER_DATA_DIR, 'nikkei_combined_5min_cleaned.csv')
TEST_CSV = os.path.join(MASTER_DATA_DIR, 'nikkei_combined_5min_cleaned.csv')  # テストデータ用
FILE_NAME_RUNNING = os.path.basename(__file__).replace('.py', '')

# 日付範囲
TRAIN_START_DATE = pd.to_datetime('2024-01-01')
TRAIN_END_DATE = pd.to_datetime('2024-11-01')
TEST_START_DATE = pd.to_datetime('2023-01-01')
TEST_END_DATE = pd.to_datetime('2024-01-01')

def prepare_data(data):
    """データの前処理"""
    print("時間帯特徴を追加中...")
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_minute'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['cos_minute'] = np.cos(2 * np.pi * data['minute'] / 60)

    print("テクニカル指標を計算中...")
    data['Returns'] = data['Close'].pct_change()
    data['Future_Return'] = (data['Close'].shift(-LOOKAHEAD_PERIOD) - data['Close']) / data['Close']
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['MACD'] = ta.macd(data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    data['BB_Upper'] = ta.bbands(data['Close'], length=20, std=2)['BBU_20_2.0']
    data['BB_Lower'] = ta.bbands(data['Close'], length=20, std=2)['BBL_20_2.0']
    data['Price_Spread'] = (data['High'] - data['Low']) / data['Close']
    data['Sentiment_Proxy'] = data['Price_Spread'] * data['Volume']

    features = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 
               'Sentiment_Proxy', 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']
    
    data = data.dropna()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    data = data.dropna()
    
    return data, features

def create_sequences(data, features):
    """シーケンスデータの作成"""
    print("シーケンスを作成中...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH - LOOKAHEAD_PERIOD):
        X.append(scaled_data[i:i + SEQUENCE_LENGTH])
        y.append(data['Future_Return'].iloc[i + SEQUENCE_LENGTH])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_model(input_shape):
    """モデルの構築"""
    print("モデルを構築中...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, X_test, y_test):
    """モデルの学習"""
    model = build_model((SEQUENCE_LENGTH, X_train.shape[2]))
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
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss_plot_{MODEL_VERSION}.png'))
    plt.close()
    
    return model

def save_model_and_scaler(model, scaler):
    """モデルとスケーラーの保存"""
    print("モデルとスケーラーを保存中...")
    save_dir = os.path.join(MODEL_DIR, f'{FILE_NAME_RUNNING}_{MODEL_VERSION}')
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(os.path.join(save_dir, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print(f"モデルを保存しました: {os.path.join(save_dir, 'lstm_model.h5')}")
    print(f"スケーラーを保存しました: {os.path.join(save_dir, 'scaler.pkl')}")

def load_model_and_scaler():
    """モデルとスケーラーのロード"""
    print("モデルとスケーラーをロード中...")
    load_dir = os.path.join(MODEL_DIR, f'{FILE_NAME_RUNNING}_{MODEL_VERSION}')
    
    model = load_model(os.path.join(load_dir, 'lstm_model.h5'))
    scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
    print(f"モデルをロードしました: {os.path.join(load_dir, 'lstm_model.h5')}")
    print(f"スケーラーをロードしました: {os.path.join(load_dir, 'scaler.pkl')}")
    
    return model, scaler

def evaluate_model(model, data, scaler, features, is_test=False):
    """モデルの評価"""
    print("予測を生成中...")
    scaled_data = scaler.transform(data[features])
    
    X = []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:i + SEQUENCE_LENGTH])
    X = np.array(X)
    
    predictions = model.predict(X, verbose=0)
    
    # 予測値をデータフレームに追加
    data['Predicted_Return'] = np.nan
    data.loc[data.index[SEQUENCE_LENGTH:len(predictions)+SEQUENCE_LENGTH], 'Predicted_Return'] = predictions.flatten()
    
    # ポジションサイジング
    data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * SCALING_FACTOR, 0, MAX_POSITION) * np.sign(data['Predicted_Return'])
    data['Position_Size'] = np.where(np.abs(data['Position_Size']) < THRESHOLD, 0, data['Position_Size'])
    data['Position_Size'] = data['Position_Size'].fillna(0)
    
    # 戦略リターン
    data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
    data['Strategy_Return'] = data['Strategy_Return'] - TRANSACTION_COST * data['Position_Size'].diff().abs()

    print(f"Strategy_Return distribution:\n{data['Strategy_Return'].describe()}")
    trades = data[data['Position_Size'].diff().abs() > 0]
    print(f"Trades:\n{trades[['Position_Size', 'Predicted_Return', 'Future_Return', 'Strategy_Return']]}")
    
    # 結果の評価
    cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod().iloc[-1]
    sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252 * 66) if data['Strategy_Return'].std() != 0 else 0
    trade_count = int(data['Position_Size'].diff().abs().gt(0).sum())
    
    print(f"\n{'テスト' if is_test else 'トレーニング'}結果:")
    print(f"累積リターン: {cumulative_return:.4f}")
    print(f"シャープレシオ: {sharpe_ratio:.4f}")
    print(f"取引回数: {trade_count}")
    
    # 予測誤差の分析
    data['Prediction_Error'] = data['Predicted_Return'] - data['Future_Return']
    error_stats = data['Prediction_Error'].describe()
    print("\n予測誤差の統計:")
    print(error_stats)
    
    # 結果の保存
    prefix = 'test_' if is_test else 'train_'
    data[['Predicted_Return', 'Future_Return', 'Prediction_Error', 'Strategy_Return']].to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}prediction_results_{MODEL_VERSION}.csv')
    )
    
    return data

def main():
    try:
        # データの読み込み
        print("データを読み込み中...")
        train_data = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])
        train_data['Datetime'] = pd.to_datetime(train_data['Datetime'], errors='coerce')
        train_data = train_data[(train_data['Datetime'] >= TRAIN_START_DATE) & 
                              (train_data['Datetime'] <= TRAIN_END_DATE)]
        train_data = train_data.set_index('Datetime')
        
        # テストデータの読み込み
        test_data = pd.read_csv(TEST_CSV, parse_dates=['Datetime'])
        test_data['Datetime'] = pd.to_datetime(test_data['Datetime'], errors='coerce')
        test_data = test_data[(test_data['Datetime'] >= TEST_START_DATE) & 
                            (test_data['Datetime'] <= TEST_END_DATE)]
        test_data = test_data.set_index('Datetime')
        
        # データの前処理
        train_data, features = prepare_data(train_data)
        test_data, _ = prepare_data(test_data)
        
        if TRAIN_MODEL:
            # シーケンスデータの作成
            X, y, scaler = create_sequences(train_data, features)
            
            # 訓練・テストデータ分割
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # モデルの学習
            model = train_model(X_train, y_train, X_test, y_test)
            
            # モデルの保存
            save_model_and_scaler(model, scaler)
        else:
            # モデルのロード
            model, scaler = load_model_and_scaler()
        
        # トレーニングデータでの評価
        train_results = evaluate_model(model, train_data, scaler, features, is_test=False)
        
        # テストデータでの評価
        test_results = evaluate_model(model, test_data, scaler, features, is_test=True)
        
        print("\n分析完了。結果はoutputディレクトリに保存されました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()