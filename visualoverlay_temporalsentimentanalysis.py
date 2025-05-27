import matplotlib.pyplot as plt

# Re-prepare percentage by year (already done earlier)
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df = df[(df['Year'] >= 2015) & (df['Year'] <= 2025)]
count_by_year = df.groupby(['Year', 'Sentiment_5Class']).size().unstack(fill_value=0)
percent_by_year = count_by_year.div(count_by_year.sum(axis=1), axis=0) * 100

# Plot with overlays
plt.figure(figsize=(12, 6))
percent_by_year.plot(kind='line', marker='o', figsize=(12, 6))

# Overlay real-world events
events = {
    2018: "Uber fatal crash",
    2020: "Tesla safety review surge",
    2021: "NHTSA Autopilot probe",
    2023: "Waymo/Cruise rollout"
}

for year, label in events.items():
    plt.axvline(x=year, color='red', linestyle='--', alpha=0.6)
    plt.text(year + 0.1, 85, label, rotation=90, verticalalignment='center', fontsize=9, color='red')

# Final touches
plt.title("Sentiment Percentage by Year (2015–2025) with Major Events")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()