import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz

# 設定
MASTER_DATA_DIR = 'master_data'
STOCKS_CSV = os.path.join(MASTER_DATA_DIR, 'daytrade_stocks.csv')

def get_tickers():
    """daytrade_stocks.csvからtickerを読み込む"""
    df = pd.read_csv(STOCKS_CSV)
    tickers = df['ticker'].tolist()
    tickers.append('NIY=F')  # 日経先物を追加
    return tickers

def get_date_range():
    """過去58日分の日付範囲を計算"""
    # 日本時間で現在時刻を取得
    jst = pytz.timezone('Asia/Tokyo')
    end_date = datetime.now(jst)
    
    # 過去58日を計算（yfinanceの制限）
    start_date = end_date - timedelta(days=58)
    
    # 日付を整形
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    print(f"データ取得期間: {start_date.strftime('%Y-%m-%d %H:%M:%S')} から {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    return start_date, end_date

def convert_to_jst(df):
    """データフレームのタイムゾーンをJSTに変換"""
    # Datetime列がタイムゾーン情報を持っている場合
    if pd.api.types.is_datetime64_any_dtype(df['Datetime']):
        # UTCからJSTに変換
        df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Tokyo')
    else:
        # タイムゾーン情報がない場合はJSTとして扱う
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('Asia/Tokyo')
    
    # タイムゾーン情報を削除して文字列に変換
    df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def download_and_save_data(ticker, start_date, end_date):
    """指定されたtickerのデータをダウンロードして保存"""
    try:
        print(f"\n{ticker} のデータをダウンロード中...")
        
        # データのダウンロード
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='5m',
            progress=False
        )
        
        if data.empty:
            print(f"警告: {ticker} のデータが取得できませんでした")
            return
        
        # マルチインデックスの場合の処理
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 必要な列のみを選択
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # インデックスをDatetime列に変換
        data = data.reset_index()
        data.rename(columns={'Datetime': 'Datetime'}, inplace=True)
        
        # タイムゾーンをJSTに変換
        data = convert_to_jst(data)
        
        # 日付範囲をファイル名に使用
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # ファイル名の作成
        filename = f"{ticker.replace('=', '_')}_{start_str}_{end_str}_5min.csv"
        filepath = os.path.join(MASTER_DATA_DIR, filename)
        
        # CSVとして保存
        data.to_csv(filepath, index=False)
        print(f"データを保存しました: {filepath}")
        print(f"データ件数: {len(data)}")
        print(f"データ期間: {data['Datetime'].iloc[0]} から {data['Datetime'].iloc[-1]}")
        
    except Exception as e:
        print(f"エラー: {ticker} のデータ取得中にエラーが発生しました: {e}")

def main():
    # マスターデータディレクトリの作成
    os.makedirs(MASTER_DATA_DIR, exist_ok=True)
    
    # tickerの取得
    tickers = get_tickers()
    print(f"取得対象の銘柄数: {len(tickers)}")
    print(f"取得対象銘柄: {', '.join(tickers)}")
    
    # 日付範囲の取得
    start_date, end_date = get_date_range()
    
    # 各tickerのデータをダウンロード
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date)
    
    print("\nダウンロードが完了しました！")

if __name__ == "__main__":
    main() 