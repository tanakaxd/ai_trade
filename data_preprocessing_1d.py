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
def fetch_yfinance_data(ticker, period="30y", interval="1d", auto_adjust=True, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=auto_adjust)
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {ticker}")
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

# 株価データ取得
ticker = "7203.T"
data = fetch_yfinance_data(ticker, period="30y", interval="1d", auto_adjust=True)
data = data[["Open", "High", "Low", "Close", "Volume"]]
data.index = pd.to_datetime(data.index).tz_localize(None)  # タイムゾーンを削除
data = data.sort_index()

# 円ドルデータ取得
usd_jpy_ticker = "JPY=X"
usd_jpy_data = fetch_yfinance_data(usd_jpy_ticker, period="30y", interval="1d", auto_adjust=True)
usd_jpy_data = usd_jpy_data[["High", "Low", "Close"]].rename(columns={"Close": "USDJPY_Close", "High": "USDJPY_High", "Low": "USDJPY_Low"})
usd_jpy_data.index = pd.to_datetime(usd_jpy_data.index).tz_localize(None)  # タイムゾーンを削除
usd_jpy_data = usd_jpy_data.sort_index()

# Volume=0のレコードを削除（休場日対応、最初に実行）
data = data[data["Volume"] != 0]

# データ結合（株価データを優先）
data = data.join(usd_jpy_data, how="left")  # 左結合で株価データのインデックスを保持

# 決算日データの読み込み
earnings_df = pd.read_csv("master_data/toyota_filing_dates.csv", parse_dates=["Filing Date"])
earnings_dates = pd.to_datetime(earnings_df["Filing Date"]).sort_values().unique()

# 取引日
trading_dates = data.index.values

# 次の決算日のインデックスを検索
idx = np.searchsorted(earnings_dates, trading_dates, side='right')

# 次の決算日を取得
next_earnings = np.full_like(trading_dates, np.datetime64('NaT'), dtype='datetime64[ns]')
valid_idx = idx < len(earnings_dates)
next_earnings[valid_idx] = earnings_dates[idx[valid_idx]]

# 次の決算日までの日数を計算
days_until_next_earnings = (next_earnings - trading_dates).astype('timedelta64[D]').astype(float)
days_until_next_earnings = np.where(np.isnan(days_until_next_earnings), 365, days_until_next_earnings)

# 決算日フラグ
is_earnings_day = np.isin(trading_dates, earnings_dates).astype(int)

# データに追加
data["days_until_next_earnings"] = pd.Series(days_until_next_earnings, index=data.index).astype(int)
data["is_earnings_day"] = pd.Series(is_earnings_day, index=data.index)

# 次の決算日までの日数をビニング
bins = [0, 5, 10, 15, 30, 60, 90, 180, 365]
data["days_until_next_earnings_cat"] = pd.cut(data["days_until_next_earnings"], bins=bins, labels=False, include_lowest=True)

# 特徴量生成
# 1. USD/JPY 特徴量
usd_jpy_features = {
    "USDJPY_pct_change": data["USDJPY_Close"].pct_change(),
    "USDJPY_sma5": data["USDJPY_Close"].rolling(window=5).mean(),
    "USDJPY_sma20": data["USDJPY_Close"].rolling(window=20).mean(),
    "USDJPY_sma100": data["USDJPY_Close"].rolling(window=100).mean(),
    "USDJPY_std_10": data["USDJPY_Close"].rolling(window=10).std(),  # 新しい特徴量
    "USDJPY_std_20": data["USDJPY_Close"].rolling(window=20).std(),  # 既存
    "USDJPY_std_50": data["USDJPY_Close"].rolling(window=50).std(),  # 新しい特徴量
    "USDJPY_std_100": data["USDJPY_Close"].rolling(window=100).std(),  # 新しい特徴量
    "USDJPY_atr": AverageTrueRange(data["USDJPY_High"], data["USDJPY_Low"], data["USDJPY_Close"], window=14).average_true_range(),
    "USDJPY_lag_1": data["USDJPY_Close"].shift(1),
    "USDJPY_lag_3": data["USDJPY_Close"].shift(3),
    "USDJPY_lag_5": data["USDJPY_Close"].shift(5),
    "USDJPY_lag_10": data["USDJPY_Close"].shift(10),
    "USDJPY_pct_change_lag_1": data["USDJPY_Close"].pct_change().shift(1),
    "USDJPY_pct_change_lag_3": data["USDJPY_Close"].pct_change().shift(3)
}
usd_jpy_df = pd.DataFrame(usd_jpy_features, index=data.index)

# 2. トレンド指標
sma_fast = SMAIndicator(data["Close"], window=5).sma_indicator()
sma_slow = SMAIndicator(data["Close"], window=20).sma_indicator()
ema_fast = EMAIndicator(data["Close"], window=5).ema_indicator()
ema_slow = EMAIndicator(data["Close"], window=20).ema_indicator()
macd = MACD(data["Close"])
macd_val = macd.macd()
macd_signal = macd.macd_signal()
macd_diff = macd.macd_diff()

# 一目均衡表
ichimoku = IchimokuIndicator(high=data["High"], low=data["Low"], window1=9, window2=26, window3=52)
ichimoku_base_line = ichimoku.ichimoku_base_line()
ichimoku_conversion_line = ichimoku.ichimoku_conversion_line()
ichimoku_a = ichimoku.ichimoku_a()
ichimoku_b = ichimoku.ichimoku_b()
ichimoku_lagging = data["Close"].shift(-26)

# ADX
adx = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
adx_val = adx.adx()
adx_pos = adx.adx_pos()
adx_neg = adx.adx_neg()

# CCI
cci = CCIIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=20)
cci_val = cci.cci()

# 3. モメンタム指標
rsi = RSIIndicator(data["Close"], window=14).rsi()
stoch = StochasticOscillator(data["High"], data["Low"], data["Close"], window=14, smooth_window=3)
stoch_k = stoch.stoch()
stoch_d = stoch.stoch_signal()

# 4. ボラティリティ指標
bb = BollingerBands(data["Close"], window=20)
bb_high = bb.bollinger_hband()
bb_low = bb.bollinger_lband()
bb_width = bb.bollinger_wband()
atr = AverageTrueRange(data["High"], data["Low"], data["Close"], window=14).average_true_range()
kc = KeltnerChannel(data["High"], data["Low"], data["Close"], window=20)
kc_high = kc.keltner_channel_hband()
kc_low = kc.keltner_channel_lband()
kc_width = kc_high - kc_low

# 5. ボリューム指標
obv = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
vwap = VolumeWeightedAveragePrice(data["High"], data["Low"], data["Close"], data["Volume"], window=14).volume_weighted_average_price()

# データに追加
data["USDJPY_Close"] = data["USDJPY_Close"]
data["USDJPY_High"] = data["USDJPY_High"]
data["USDJPY_Low"] = data["USDJPY_Low"]
data["USDJPY_pct_change"] = usd_jpy_df["USDJPY_pct_change"]
data["USDJPY_sma5"] = usd_jpy_df["USDJPY_sma5"]
data["USDJPY_sma20"] = usd_jpy_df["USDJPY_sma20"]
data["USDJPY_sma100"] = usd_jpy_df["USDJPY_sma100"]
data["USDJPY_std_10"] = usd_jpy_df["USDJPY_std_10"]
data["USDJPY_std_20"] = usd_jpy_df["USDJPY_std_20"]
data["USDJPY_std_50"] = usd_jpy_df["USDJPY_std_50"]
data["USDJPY_std_100"] = usd_jpy_df["USDJPY_std_100"]
data["USDJPY_atr"] = usd_jpy_df["USDJPY_atr"]
data["USDJPY_lag_1"] = usd_jpy_df["USDJPY_lag_1"]
data["USDJPY_lag_3"] = usd_jpy_df["USDJPY_lag_3"]
data["USDJPY_lag_5"] = usd_jpy_df["USDJPY_lag_5"]
data["USDJPY_lag_10"] = usd_jpy_df["USDJPY_lag_10"]
data["USDJPY_pct_change_lag_1"] = usd_jpy_df["USDJPY_pct_change_lag_1"]
data["USDJPY_pct_change_lag_3"] = usd_jpy_df["USDJPY_pct_change_lag_3"]
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

# 6. ラグ特徴量
lag_features = {}
for lag in [1, 3, 5, 10, 15, 20, 30]:
    lag_features[f"Close_lag_{lag}"] = data["Close"].shift(lag)
    lag_features[f"Volume_lag_{lag}"] = data["Volume"].shift(lag)
    lag_features[f"Close_pct_change_{lag}"] = data["Close"].pct_change(periods=lag)
lag_df = pd.DataFrame(lag_features, index=data.index)

# 7. 統計量
stat_features = {}
for window in [5, 10, 20, 30, 50, 100]:
    stat_features[f"Close_mean_{window}"] = data["Close"].rolling(window=window).mean()
    stat_features[f"Close_std_{window}"] = data["Close"].rolling(window=window).std()
    stat_features[f"Volume_mean_{window}"] = data["Volume"].rolling(window=window).mean()
stat_df = pd.DataFrame(stat_features, index=data.index)

# ラグ特徴量と統計量を先に結合
data = pd.concat([data, lag_df, stat_df], axis=1)

# 8. 寄り/引け特徴量
open_close_features = {
    "Open_to_prev_close": data["Open"] - data["Close"].shift(1),
    "Open_to_sma5": data["Open"] - data["sma_fast"],
    "Open_volatility": data["Open"].rolling(window=5).std(),
    "Close_to_sma5": data["Close"] - data["sma_fast"],
    "Close_to_vwap": data["Close"] - data["vwap"],
    "Close_volatility": data["Close"].rolling(window=5).std()
}
open_close_df = pd.DataFrame(open_close_features, index=data.index)

# 9. 特徴量合成
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

# 10. 乖離/インバランス
imbalance_features = {
    "Close_Open_diff": data["Close"] - data["Open"],
    "High_Low_diff": data["High"] - data["Low"],
    "Close_SMA5_diff": data["Close"] - data["sma_fast"],
    "Buy_Sell_Imbalance": (data["Close"] - data["Open"]) / (data["High"] - data["Low"]).replace(0, np.nan),
    "Volume_Imbalance": data["Volume"] / data["Volume_mean_5"],
    "RSI_MACD_Imbalance": data["rsi"] / data["macd"].replace(0, np.nan)
}
imbalance_df = pd.DataFrame(imbalance_features, index=data.index)

# 11. カテゴリカル変数
cat_features = {
    "day_of_week": data.index.dayofweek,
    "month": data.index.month,
    "quarter": ((data.index.month - 4) % 12 // 3) + 1,  # トヨタの3月期決算
    "is_month_end": (data.index.is_month_end).astype(int)
}
cat_df = pd.DataFrame(cat_features, index=data.index)

# カテゴリカル変数のリスト
cat_columns = ["day_of_week", "month", "quarter", "is_month_end", "is_earnings_day", "days_until_next_earnings_cat"]

# 結合前の列名衝突チェック
existing_columns = data.columns
new_columns = cat_df.columns
conflicting_columns = [col for col in new_columns if col in existing_columns]
if conflicting_columns:
    print(f"Warning: Potential column conflicts before concat: {conflicting_columns}")

# 残りの特徴量を結合
data = pd.concat([data, open_close_df, synth_df, imbalance_df, cat_df], axis=1)

# 結合後の重複列チェック
duplicated_columns = data.columns[data.columns.duplicated()].tolist()
if duplicated_columns:
    print(f"Warning: Duplicated columns found: {duplicated_columns}")
    data = data.loc[:, ~data.columns.duplicated(keep='first')]
    print("Duplicated columns removed.")

# 異常値チェックと処理
data = data.replace([np.inf, -np.inf], np.nan)
data = data.iloc[103:]  # 初期103レコードをカット（ユーザーの要望）
data = data.fillna(method="ffill")  # 前方補完（USD/JPY の欠損値も補完）

# 特徴量リスト（入力価格データとUSD/JPY生データを除外）
columns_to_drop = ["Open", "High", "Low", "Close", "Volume", "USDJPY_Close", "USDJPY_High", "USDJPY_Low"]
features = data.drop(columns=columns_to_drop).columns
print(f"Total features generated: {len(features)}")

# ターゲット（Close_i+5 の相対変化）
y = pd.DataFrame({
    "Close_i+5_ratio": data["Close"].shift(-5) / data["Close"]  # Close_i+5 / Close_current
}).dropna()
X = data[features].loc[y.index]

# 現在の終値（後で予測値を絶対値に戻す用）
Close_current = data["Close"].loc[X.index]

# データ保存（スケーリングなし）
data_dict = {
    "X": X,
    "y_orig": y,
    "features": features,
    "Close_current": Close_current,
    "cat_columns": cat_columns
}
pd.to_pickle(data_dict, "data/processed_data_1d.pkl")

print(data)
data.to_csv("data/processed_data_1d.csv")