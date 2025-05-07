import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
data_dict = pd.read_pickle("data/processed_data.pkl")
X = data_dict["X"]
y_orig = data_dict["y_orig"]["Close_i+5"]
close_current = data_dict["Close_current"]

# インデックスを統一
data = X.copy()
data["Close_current"] = close_current
data["Close_i+5"] = y_orig

# 全体の時系列プロット
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Close_current"], label="Close_current")
plt.plot(data.index, data["Close_i+5"], label="Close_i+5", alpha=0.5)
plt.title("Time Series of Close_current and Close_i+5")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("output/time_series_plot.png")
plt.close()

# フォールド5のテストデータ範囲を特定（仮定: データの末尾20%をテストとする）
test_size = int(len(data) * 0.2)
test_data = data.iloc[-test_size:]
test_y_orig = y_orig.iloc[-test_size:]

# フォールド5のプロット
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data["Close_current"], label="Close_current (Fold 5 Test)")
plt.plot(test_data.index, test_y_orig, label="Close_i+5 (Fold 5 Test)", alpha=0.5)
plt.title("Fold 5 Test Data: Close_current and Close_i+5")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("output/fold5_test_plot.png")
plt.close()

# 異常値チェック（ボックスプロット）
plt.figure(figsize=(8, 4))
plt.boxplot([data["Close_current"], y_orig], labels=["Close_current", "Close_i+5"])
plt.title("Box Plot of Close_current and Close_i+5")
plt.savefig("output/box_plot.png")
plt.close()