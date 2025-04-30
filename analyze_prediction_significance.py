import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
# plt.rcParams['font.family'] = 'Hiragino Sans'  # Mac用
# plt.rcParams['font.family'] = 'IPAexGothic'  # Linux用

# 設定
OUTPUT_DIR = 'output'
PREDICTION_ERRORS_FILE = os.path.join(OUTPUT_DIR, 'prediction_errors.csv')

def load_data():
    """データの読み込み"""
    print("データを読み込み中...")
    df = pd.read_csv(PREDICTION_ERRORS_FILE, parse_dates=['Datetime'], index_col='Datetime')
    return df

def analyze_correlation(df):
    """相関分析"""
    print("\n=== 相関分析 ===")
    correlation = df['Predicted_Return'].corr(df['Future_Return'])
    print(f"予測値と実際のリターンの相関係数: {correlation:.4f}")
    
    # 散布図のプロット
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Predicted_Return', y='Future_Return', data=df, scatter_kws={'alpha':0.3})
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_correlation.png'))
    plt.close()

def analyze_prediction_error(df):
    """予測誤差の分析"""
    print("\n=== 予測誤差の分析 ===")
    error = df['Prediction_Error']
    
    # 基本統計量
    print("予測誤差の基本統計量:")
    print(error.describe())
    
    # 正規性の検定
    _, p_value = stats.normaltest(error.dropna())  # NaNを除外
    print(f"\n予測誤差の正規性検定 p値: {p_value:.4f}")
    
    # 誤差の分布プロット
    plt.figure(figsize=(10, 6))
    sns.histplot(error, kde=True)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_error_distribution.png'))
    plt.close()

def analyze_direction_accuracy(df):
    """方向性の正解率分析"""
    print("\n=== 方向性の正解率分析 ===")
    
    # 方向性の判定
    df['Predicted_Direction'] = np.where(df['Predicted_Return'] > 0, 1, 0)
    df['Actual_Direction'] = np.where(df['Future_Return'] > 0, 1, 0)
    
    # 混同行列
    cm = confusion_matrix(df['Actual_Direction'], df['Predicted_Direction'])
    print("混同行列:")
    print(cm)
    
    # 分類レポート
    print("\n分類レポート:")
    print(classification_report(df['Actual_Direction'], df['Predicted_Direction']))
    
    # 正解率
    accuracy = (df['Predicted_Direction'] == df['Actual_Direction']).mean()
    print(f"\n方向性の正解率: {accuracy:.4f}")

def analyze_return_distributions(df):
    """リターンの分布比較"""
    print("\n=== リターンの分布比較 ===")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Predicted_Return'], label='Predicted Return')
    sns.kdeplot(df['Future_Return'], label='Actual Return')
    plt.title('Return Distributions Comparison')
    plt.xlabel('Return')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'return_distributions.png'))
    plt.close()
    
    # 分布の統計的比較
    _, p_value = stats.ks_2samp(df['Predicted_Return'].dropna(), df['Future_Return'].dropna())
    print(f"Kolmogorov-Smirnov検定 p値: {p_value:.4f}")

def analyze_time_dependent_performance(df):
    """時間依存的なパフォーマンス分析"""
    print("\n=== 時間依存的なパフォーマンス分析 ===")
    
    # 日付ごとの正解率
    daily_accuracy = df.groupby(df.index.date).apply(
        lambda x: (x['Predicted_Direction'] == x['Actual_Direction']).mean()
    )
    
    # 日付インデックスをDatetimeIndexに変換
    daily_accuracy.index = pd.to_datetime(daily_accuracy.index)
    
    plt.figure(figsize=(12, 6))
    daily_accuracy.plot()
    plt.title('Daily Accuracy Trend')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Prediction')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_accuracy.png'))
    plt.close()
    
    # 月次平均正解率
    monthly_accuracy = daily_accuracy.resample('M').mean()
    print("\n月次平均正解率:")
    print(monthly_accuracy)

def main():
    # データの読み込み
    df = load_data()
    
    # 各種分析の実行
    analyze_correlation(df)
    analyze_prediction_error(df)
    analyze_direction_accuracy(df)
    analyze_return_distributions(df)
    analyze_time_dependent_performance(df)
    
    print("\n分析完了。結果はoutputディレクトリに保存されました。")

if __name__ == "__main__":
    main() 