import pandas as pd
import matplotlib.pyplot as plt

# Load the 5-class labeled dataset with dates
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")

# Convert date to datetime and extract year
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year

# Filter data within the 2015–2025 range
df = df[(df['Year'] >= 2015) & (df['Year'] <= 2025)]

# Count of sentiments per year
count_by_year = df.groupby(['Year', 'Sentiment_5Class']).size().unstack(fill_value=0)

# Percentage of sentiments per year
percent_by_year = count_by_year.div(count_by_year.sum(axis=1), axis=0) * 100

# Plot sentiment counts over time
plt.figure(figsize=(12, 6))
count_by_year.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 6))
plt.title("Sentiment Count by Year (2015–2025)")
plt.xlabel("Year")
plt.ylabel("Article Count")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot sentiment percentages over time
plt.figure(figsize=(12, 6))
percent_by_year.plot(kind='line', marker='o', figsize=(12, 6))
plt.title("Sentiment Percentage by Year (2015–2025)")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()