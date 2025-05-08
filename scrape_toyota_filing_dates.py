import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime

# 対象URLリスト
URLS = [
    "https://global.toyota/jp/ir/financial-results/",
    "https://global.toyota/jp/ir/financial-results/archives/01.html",
    "https://global.toyota/jp/ir/financial-results/archives/02.html",
    "https://global.toyota/jp/ir/financial-results/archives/03.html"
]

# 日本語日付をYYYY-MM-DD形式に変換
def parse_japanese_date(date_str):
    try:
        # 正規表現で「YYYY年MM月DD日」を抽出（MMやDDは1桁でも対応）
        match = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str)
        if not match:
            return None
        year, month, day = match.groups()
        # 月と日を2桁にゼロ埋め
        formatted_date = f"{year}-{int(month):02d}-{int(day):02d}"
        # 日付の妥当性をチェック
        datetime.strptime(formatted_date, "%Y-%m-%d")
        return formatted_date
    except (ValueError, AttributeError):
        return None

# 1つのURLから決算日を抽出
def scrape_filing_dates(url):
    try:
        # ページを取得
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # HTMLを解析
        soup = BeautifulSoup(response.text, "html.parser")
        
        # <h3>タグを検索
        dates = []
        for h3 in soup.find_all("h3"):
            text = h3.get_text(strip=True)
            if "決算情報" in text:
                # カッコ内の日付を抽出
                match = re.search(r"（(\d{4}年\d{1,2}月\d{1,2}日)）", text)
                if match:
                    date_str = match.group(1)
                    formatted_date = parse_japanese_date(date_str)
                    if formatted_date:
                        dates.append(formatted_date)
        
        return dates
    except requests.exceptions.RequestException as e:
        print(f"エラー: {url} の取得に失敗しました - {e}")
        return []
    except Exception as e:
        print(f"エラー: {url} の処理中に問題が発生しました - {e}")
        return []

# メイン処理
def main():
    all_dates = []
    
    # 各URLからデータを取得
    for url in URLS:
        print(f"{url} からデータを取得中...")
        dates = scrape_filing_dates(url)
        if dates:
            print(f"取得した日付: {dates}")
            all_dates.extend(dates)
        else:
            print(f"{url} から日付を抽出できませんでした。")
    
    # 重複を排除してソート
    unique_dates = sorted(list(set(all_dates)))
    
    if unique_dates:
        # DataFrameに変換
        df = pd.DataFrame(unique_dates, columns=["Filing Date"])
        print("\nトヨタの決算日（提出日）:")
        print(df)
        
        # CSVに保存
        df.to_csv("toyota_filing_dates.csv", index=False)
        print("データが 'toyota_filing_dates.csv' に保存されました。")
    else:
        print("すべてのページから日付を抽出できませんでした。")

if __name__ == "__main__":
    main()