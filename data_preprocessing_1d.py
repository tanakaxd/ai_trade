import pandas as pd
import numpy as np
import yfinance as yf
import time
from yfinance.exceptions import YFRateLimitError
import os
from ta.trend import SMAIndicator, MACD, EMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.trend import CCIIndicator

# ディレクトリ作成
os.makedirs("data", exist_ok=True)

# データ取得（yfinance）
def fetch_yfinance_data(ticker, period="30y", interval="1d", auto_adjust=True, retries=3, wait_time=5, cache_file="data/raw_data.csv"):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return data
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=auto_adjust)
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {ticker}")
            data.to_csv(cache_file)
            return data
        except YFRateLimitError:
            if attempt < retries - 1:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to fetch data for {ticker} after {retries} attempts due to rate limiting.")
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error fetching data: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to fetch data for {ticker} after {retries} attempts: {e}")

ticker = "7203.T"
data = fetch_yfinance_data(ticker, period="30y", interval="1d", auto_adjust=True)
data = data[["Open", "High", "Low", "Close", "Volume"]]
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Volume=0のレコードを削除（休場日対応、特徴量計算前に実行）
data = data[data["Volume"] != 0]

# 特徴量生成
# 1. トレンド指標
sma_fast = SMAIndicator(data["Close"], window=5).sma_indicator()
sma_slow = SMAIndicator(data["Close"], window=20).sma_indicator()
ema_fast = EMAIndicator(data["Close"], window=5).ema_indicator()
ema_slow = EMAIndicator(data["Close"], window=20).ema_indicator()
macd = MACD(data["Close"])
macd_val = macd.macd()
macd_signal = macd.macd_signal()
macd_diff = macd.macd_diff()

# 新規追加: 一目均衡表 (Ichimoku Cloud)
ichimoku = IchimokuIndicator(high=data["High"], low=data["Low"], window1=9, window2=26, window3=52)
ichimoku_base_line = ichimoku.ichimoku_base_line()  # 基準線
ichimoku_conversion_line = ichimoku.ichimoku_conversion_line()  # 転換線
ichimoku_a = ichimoku.ichimoku_a()  # 先行スパン1
ichimoku_b = ichimoku.ichimoku_b()  # 先行スパン2
ichimoku_lagging = data["Close"].shift(-26)  # 遅行スパン（taでは計算されないため手動で）

# 新規追加: ADX (Average Directional Index)
adx = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
adx_val = adx.adx()
adx_pos = adx.adx_pos()  # +DI
adx_neg = adx.adx_neg()  # -DI

# 新規追加: CCI (Commodity Channel Index)
cci = CCIIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=20)
cci_val = cci.cci()

# 2. モメンタム指標
rsi = RSIIndicator(data["Close"], window=14).rsi()
stoch = StochasticOscillator(data["High"], data["Low"], data["Close"], window=14, smooth_window=3)
stoch_k = stoch.stoch()
stoch_d = stoch.stoch_signal()

# 3. ボラティリティ指標
bb = BollingerBands(data["Close"], window=20)
bb_high = bb.bollinger_hband()
bb_low = bb.bollinger_lband()
bb_width = bb.bollinger_wband()
atr = AverageTrueRange(data["High"], data["Low"], data["Close"], window=14).average_true_range()
kc = KeltnerChannel(data["High"], data["Low"], data["Close"], window=20)
kc_high = kc.keltner_channel_hband()
kc_low = kc.keltner_channel_lband()
kc_width = kc_high - kc_low

# 4. ボリューム指標
obv = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
vwap = VolumeWeightedAveragePrice(data["High"], data["Low"], data["Close"], data["Volume"], window=14).volume_weighted_average_price()

# データに追加
data["sma_fast"] = sma_fast
data["sma_slow"] = sma_slow
data["ema_fast"] = ema_fast
data["ema_slow"] = ema_slow
data["macd"] = macd_val
data["macd_signal"] = macd_signal
data["macd_diff"] = macd_diff
data["ichimoku_base_line"] = ichimoku_base_line
data["ichimoku_conversion_line"] = ichimoku_conversion_line
data["ichimoku_a"] = ichimoku_a
data["ichimoku_b"] = ichimoku_b
data["ichimoku_lagging"] = ichimoku_lagging
data["adx"] = adx_val
data["adx_pos"] = adx_pos
data["adx_neg"] = adx_neg
data["cci"] = cci_val
data["rsi"] = rsi
data["stoch_k"] = stoch_k
data["stoch_d"] = stoch_d
data["bb_high"] = bb_high
data["bb_low"] = bb_low
data["bb_width"] = bb_width
data["atr"] = atr
data["kc_high"] = kc_high
data["kc_low"] = kc_low
data["kc_width"] = kc_width
data["obv"] = obv
data["vwap"] = vwap

# 5. ラグ特徴量（拡張）
lag_features = {}
for lag in [1, 3, 5, 10, 15, 20, 30]:
    lag_features[f"Close_lag_{lag}"] = data["Close"].shift(lag)
    lag_features[f"Volume_lag_{lag}"] = data["Volume"].shift(lag)
    lag_features[f"Close_pct_change_{lag}"] = data["Close"].pct_change(periods=lag)
lag_df = pd.DataFrame(lag_features, index=data.index)

# 6. 統計量（拡張）
stat_features = {}
for window in [5, 10, 20, 30, 50, 100]:
    stat_features[f"Close_mean_{window}"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}"] = data["Volume"].rolling(window=window).mean()
stat_df = pd.DataFrame(stat_features, index=data.index)

# ラグ特徴量と統計量を結合
data = pd.concat([data, lag_df, stat_df], axis=1)

# 7. 寄り/引け特徴量
open_close_features = {
    "Open_to_prev_close": data["Open"] - data["Close"].shift(1),
    "Open_to_sma5": data["Open"] - data["sma_fast"],
    "Open_volatility": data["Open"].rolling(window=5).std(),
    "Close_to_sma5": data["Close"] - data["sma_fast"],
    "Close_to_vwap": data["Close"] - data["vwap"],
    "Close_volatility": data["Close"].rolling(window=5).std()
}
open_close_df = pd.DataFrame(open_close_features, index=data.index)

# 8. 特徴量合成（拡張）
synth_features = {
    "Close_lag_1_plus_Volume_lag_1": data["Close_lag_1"] + data["Volume_lag_1"],
    "Close_lag_1_div_Volume_lag_1": data["Close_lag_1"] / data["Volume_lag_1"].replace(0, np.nan),
    "RSI_minus_MACD": data["rsi"] - data["macd"],
    "Close_mean_5_plus_BBW": data["Close_mean_5"] + data["bb_width"],
    "Close_lag_5_div_SMA5": data["Close_lag_5"] / data["sma_fast"].replace(0, np.nan),
    "Close_lag_1_times_Volume_lag_1": data["Close_lag_1"] * data["Volume_lag_1"],
    "Close_lag_3_div_Volume_lag_3": data["Close_lag_3"] / data["Volume_lag_3"].replace(0, np.nan),
    "RSI_plus_MACD": data["rsi"] + data["macd"],
    "BBW_times_VWAP": data["bb_width"] * data["vwap"]
}
synth_df = pd.DataFrame(synth_features, index=data.index)

# 9. 乖離/インバランス
imbalance_features = {
    "Close_Open_diff": data["Close"] - data["Open"],
    "High_Low_diff": data["High"] - data["Low"],
    "Close_SMA5_diff": data["Close"] - data["sma_fast"],
    "Buy_Sell_Imbalance": (data["Close"] - data["Open"]) / (data["High"] - data["Low"]).replace(0, np.nan),
    "Volume_Imbalance": data["Volume"] / data["Volume_mean_5"],
    "RSI_MACD_Imbalance": data["rsi"] / data["macd"].replace(0, np.nan)
}
imbalance_df = pd.DataFrame(imbalance_features, index=data.index)

# 10. カテゴリカル変数
cat_features = {
    "day_of_week": data.index.dayofweek,
}
cat_df = pd.DataFrame(cat_features, index=data.index)

# 11. センチメント（ダミー）
sentiment = pd.Series(np.random.uniform(-1, 1, len(data)), index=data.index, name="sentiment")

# 全ての特徴量を結合
data = pd.concat([data, open_close_df, synth_df, imbalance_df, cat_df, sentiment], axis=1)

# 異常値チェックと処理
data = data.replace([np.inf, -np.inf], np.nan)
data = data.iloc[100:]  # 初期100レコードをカット
data = data.fillna(method="ffill")  # 前方補完

# 特徴量リスト（入力価格データを除外）
columns_to_drop = ["Open", "High", "Low", "Close", "Volume"]
features = data.drop(columns=columns_to_drop).columns
print(f"Total features generated: {len(features)}")

# ターゲット（Close_i+5）
y = pd.DataFrame({"Close_i+5": data["Close"].shift(-5)}).dropna()
X = data[features].loc[y.index]

# 現在の終値
Close_current = data["Close"].loc[X.index]

# データ保存（スケーリングなし）
data_dict = {
    "X": X,
    "y_orig": y,
    "features": features,
    "Close_current": Close_current
}
pd.to_pickle(data_dict, "data/processed_data_1d.pkl")

print(data)
data.to_csv("data/processed_data_1d.csv")