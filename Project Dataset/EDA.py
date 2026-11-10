from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "Raw Data"
OUTPUT = BASE_DIR / "eda_output"
OUTPUT.mkdir(exist_ok=True)

files = {
    "EURUSD": "EURUSD-10-year-daily-data.csv",
    "GBPUSD": "GBPUSD-10-year-daily-data.csv",
    "USDJPY": "USDJPY-10-year-daily-data.csv",
    "USDCHF": "USDCHF-10-year-daily-data.csv",
    "AUDUSD": "AUDUSD-10-year-daily-data.csv",
    "USDCAD": "USDCAD-10-year-daily-data.csv",
    "NZDUSD": "NZDUSD-10-year-daily-data.csv",
}

summary = []
close_data = []

for pair, filename in files.items():
    df = pd.read_csv(RAW_DATA / filename)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    summary.append({
        "Pair": pair,
        "Rows": len(df),
        "Start Date": df["time"].min(),
        "End Date": df["time"].max(),
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Dates": df["time"].duplicated().sum(),
        "Minimum Close": df["close"].min(),
        "Maximum Close": df["close"].max(),
    })

    close_data.append(df[["time", "close"]].rename(columns={"close": pair}))

# EDA summary table
summary_df = pd.DataFrame(summary)
print(summary_df)
summary_df.to_csv(OUTPUT / "eda_summary.csv", index=False)

# Merge closing prices by common date
merged = close_data[0]

for data in close_data[1:]:
    merged = merged.merge(data, on="time", how="inner")

merged.to_csv(OUTPUT / "merged_close_prices.csv", index=False)

# Daily returns
returns = merged.set_index("time").pct_change().dropna()
returns.to_csv(OUTPUT / "daily_returns.csv")

# Correlation matrix
correlation = returns.corr()
print("\nCorrelation Matrix:")
print(correlation)
correlation.to_csv(OUTPUT / "correlation_matrix.csv")

# Normalized price chart
normalized_prices = merged.set_index("time")
normalized_prices = normalized_prices / normalized_prices.iloc[0] * 100

plt.figure(figsize=(14, 6))
for pair in files:
    plt.plot(normalized_prices.index, normalized_prices[pair], label=pair)

plt.title("Normalized FX Pair Closing Prices")
plt.xlabel("Date")
plt.ylabel("Indexed Price (Start = 100)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUTPUT / "normalized_price_chart.png")
plt.show()
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
plt.title("Daily Return Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT / "correlation_heatmap.png")
plt.show()
plt.close()
