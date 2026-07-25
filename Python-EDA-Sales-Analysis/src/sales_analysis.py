import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/sales_dataset.csv")

# -----------------------------
# Display Dataset
# -----------------------------
print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Summary Statistics")
print("=" * 50)
print(df.describe())

# -----------------------------
# Total Sales
# -----------------------------
total_sales = df["Sales"].sum()
print("\nTotal Sales:", total_sales)

# -----------------------------
# Total Profit
# -----------------------------
total_profit = df["Profit"].sum()
print("Total Profit:", total_profit)

# -----------------------------
# Sales by Region
# -----------------------------
region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8,5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("../images/region_sales.png")
plt.show()

# -----------------------------
# Sales by Category
# -----------------------------
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar", color="orange")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("../images/category_sales.png")
plt.show()

# -----------------------------
# Monthly Sales Trend
# -----------------------------
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
    .reindex(month_order)
)

plt.figure(figsize=(8,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/monthly_sales.png")
plt.show()

# -----------------------------
# Profit by Region
# -----------------------------
region_profit = df.groupby("Region")["Profit"].sum()

plt.figure(figsize=(7,7))
region_profit.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Profit Distribution by Region")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../images/profit_distribution.png")
plt.show()

# -----------------------------
# Profit Margin
# -----------------------------
df["Profit Margin (%)"] = (
    df["Profit"] / df["Sales"]
) * 100

print("\nProfit Margin")
print(df[["Month", "Region", "Profit Margin (%)"]])

# -----------------------------
# Best Region
# -----------------------------
best_region = region_sales.idxmax()
print("\nBest Performing Region:", best_region)

# -----------------------------
# Best Category
# -----------------------------
best_category = category_sales.idxmax()
print("Best Selling Category:", best_category)

# -----------------------------
# Highest Sale
# -----------------------------
highest_sale = df.loc[df["Sales"].idxmax()]

print("\nHighest Sales Record")
print(highest_sale)

# -----------------------------
# Save Clean Dataset
# -----------------------------
df.to_csv("../data/cleaned_sales_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")

print("\nEDA Analysis Completed Successfully!")