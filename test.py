import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# FMP APIキーを環境変数から取得
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise ValueError("エラー: FMP_API_KEY環境変数が設定されていません。")

# トヨタのティッカーシンボル
SYMBOL = "TM"

# FMP List of Dates APIエンドポイント
BASE_URL = "https://financialmodelingprep.com/api/v3/cash-flow-statement/"

# APIリクエストを送信してデータを取得
def get_financial_statement_dates(symbol, api_key):
    url = f"{BASE_URL}{symbol}?apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # エラーがあれば例外を発生
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"エラー: APIリクエストに失敗しました - {e}")
        return None

# 決算日を抽出して整理
def extract_filing_dates(data):
    if not data:
        return None
    
    # 決算日（filing date）を抽出
    dates = []
    for statement in data:
        if "date" in statement:
            filing_date = statement["date"]
            dates.append(filing_date)
    
    # 日付をソートしてユニークなものだけにする
    dates = sorted(list(set(dates)))
    return dates

# 過去30年間のデータをフィルタリング
def filter_last_30_years(dates):
    if not dates:
        return None
    
    # 現在の日付から30年前を計算
    today = datetime.today()
    thirty_years_ago = today - timedelta(days=30*365)
    
    # 30年以内の日付だけをフィルタリング
    filtered_dates = [date for date in dates if datetime.strptime(date, "%Y-%m-%d") >= thirty_years_ago]
    return filtered_dates

# メイン処理
def main():
    # データ取得
    data = get_financial_statement_dates(SYMBOL, API_KEY)
    
    if data:
        # 決算日を抽出
        filing_dates = extract_filing_dates(data)
        
        if filing_dates:
            # 過去30年間の決算日をフィルタリング
            filtered_dates = filter_last_30_years(filing_dates)
            
            if filtered_dates:
                # 結果をDataFrameに変換して表示
                df = pd.DataFrame(filtered_dates, columns=["Filing Date"])
                print(f"トヨタ (TM) の過去30年間の決算日（提出日）:")
                print(df)
                
                # CSVに保存（オプション）
                df.to_csv("toyota_filing_dates.csv", index=False)
                print("データが 'toyota_filing_dates.csv' に保存されました。")
            else:
                print("過去30年間のデータが見つかりませんでした。")
        else:
            print("決算日データが見つかりませんでした。")
    else:
        print("データの取得に失敗しました。")

if __name__ == "__main__":
    main()