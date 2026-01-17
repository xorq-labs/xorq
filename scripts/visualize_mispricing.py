"""
Visualize diamond mispricing analysis results.

Usage: python scripts/visualize_mispricing.py
"""

import subprocess

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


# Run xorq to get the data
print("Generating mispricing data...")
subprocess.run(
    ["xorq", "run", "diamonds-mispricing", "-o", "/tmp/mispricing_full.parquet"],
    check=True,
)


df = pq.read_table("/tmp/mispricing_full.parquet").to_pandas()

print(f"Loaded {len(df)} diamonds")
print("\nPrice Category Distribution:")
print(df["price_category"].value_counts())

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Predicted vs Actual scatter with deal score coloring
ax1 = axes[0, 0]
scatter = ax1.scatter(
    df["price"],
    df["predicted_price"],
    c=df["deal_score"],
    cmap="RdYlGn",  # Red = overpriced, Green = underpriced
    alpha=0.5,
    s=20,
)
ax1.plot(
    [df["price"].min(), df["price"].max()],
    [df["price"].min(), df["price"].max()],
    "k--",
    alpha=0.3,
    label="Perfect prediction",
)
ax1.set_xlabel("Actual Price ($)", fontsize=12)
ax1.set_ylabel("Predicted Price ($)", fontsize=12)
ax1.set_title(
    "Price Prediction vs Actual (colored by deal score)", fontsize=14, fontweight="bold"
)
plt.colorbar(scatter, ax=ax1, label="Deal Score")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Deal score distribution by category
ax2 = axes[0, 1]
for category, color in [
    ("underpriced", "green"),
    ("fair_priced", "gray"),
    ("overpriced", "red"),
]:
    data = df[df["price_category"] == category]["deal_score"]
    ax2.hist(data, bins=50, alpha=0.6, label=category, color=color)
ax2.set_xlabel("Deal Score", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Deal Score Distribution by Category", fontsize=14, fontweight="bold")
ax2.legend()
ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.3)

# 3. Arbitrage score by carat (for underpriced only)
ax3 = axes[1, 0]
underpriced = (
    df[df["price_category"] == "underpriced"]
    .sort_values("arbitrage_score", ascending=False)
    .head(500)
)
scatter2 = ax3.scatter(
    underpriced["carat"],
    underpriced["arbitrage_score"],
    c=underpriced["clarity_score"],
    cmap="viridis",
    alpha=0.6,
    s=50,
)
ax3.set_xlabel("Carat", fontsize=12)
ax3.set_ylabel("Arbitrage Score", fontsize=12)
ax3.set_title(
    "Top 500 Arbitrage Opportunities by Carat", fontsize=14, fontweight="bold"
)
plt.colorbar(scatter2, ax=ax3, label="Clarity Score")
ax3.grid(True, alpha=0.3)

# 4. Price vs Quality Score patterns
ax4 = axes[1, 1]
df["total_quality"] = df["cut_score"] + df["color_score"] + df["clarity_score"]
for category, color, marker in [
    ("underpriced", "green", "o"),
    ("fair_priced", "gray", "."),
    ("overpriced", "red", "x"),
]:
    subset = df[df["price_category"] == category].sample(
        min(1000, len(df[df["price_category"] == category]))
    )
    ax4.scatter(
        subset["total_quality"],
        subset["price"],
        c=color,
        alpha=0.4,
        s=10,
        marker=marker,
        label=category,
    )
ax4.set_xlabel("Total Quality Score (cut + color + clarity)", fontsize=12)
ax4.set_ylabel("Actual Price ($)", fontsize=12)
ax4.set_title("Price vs Quality - Mispricing Patterns", fontsize=14, fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/diamond_mispricing_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Visualization saved to: /tmp/diamond_mispricing_analysis.png")

# Print key insights
print("\n" + "=" * 70)
print("KEY INSIGHTS - DIAMOND ARBITRAGE OPPORTUNITIES")
print("=" * 70)

print("\n1. MARKET OVERVIEW:")
print(f"   Total diamonds analyzed: {len(df):,}")
print(
    f"   Underpriced (>20% below predicted): {len(df[df['price_category'] == 'underpriced']):,} ({100 * len(df[df['price_category'] == 'underpriced']) / len(df):.1f}%)"
)
print(
    f"   Overpriced (>20% above predicted): {len(df[df['price_category'] == 'overpriced']):,} ({100 * len(df[df['price_category'] == 'overpriced']) / len(df):.1f}%)"
)
print(
    f"   Fair priced: {len(df[df['price_category'] == 'fair_priced']):,} ({100 * len(df[df['price_category'] == 'fair_priced']) / len(df):.1f}%)"
)

print("\n2. BEST ARBITRAGE OPPORTUNITIES (Top 10):")
top_deals = df[df["price_category"] == "underpriced"].nlargest(10, "arbitrage_score")
for idx, (i, row) in enumerate(top_deals.iterrows(), 1):
    print(
        f"   #{idx}: {row['carat']:.2f}ct @ ${row['price']:,.0f} "
        f"(predicted: ${row['predicted_price']:,.0f}, "
        f"deal score: {row['deal_score']:.1f}%)"
    )

print("\n3. PATTERN ANALYSIS:")
underpriced_df = df[df["price_category"] == "underpriced"]
print(f"   Avg underpriced carat: {underpriced_df['carat'].mean():.2f}ct")
print(f"   Avg underpriced price: ${underpriced_df['price'].mean():,.0f}")
print(f"   Avg predicted price: ${underpriced_df['predicted_price'].mean():,.0f}")
print(
    f"   Avg savings opportunity: ${(underpriced_df['predicted_price'] - underpriced_df['price']).mean():,.0f}"
)

overpriced_df = df[df["price_category"] == "overpriced"]
print(f"\n   Avg overpriced carat: {overpriced_df['carat'].mean():.2f}ct")
print(f"   Avg overpriced price: ${overpriced_df['price'].mean():,.0f}")
print(f"   Avg predicted price: ${overpriced_df['predicted_price'].mean():,.0f}")
print(
    f"   Avg premium paid: ${(overpriced_df['price'] - overpriced_df['predicted_price']).mean():,.0f}"
)

print("\n4. FEATURE PATTERNS:")
print("\n   UNDERPRICED diamonds (good deals) tend to have:")
print(f"     Cut score: {underpriced_df['cut_score'].mean():.2f} (0=Fair, 4=Ideal)")
print(f"     Color score: {underpriced_df['color_score'].mean():.2f} (0=J, 6=D)")
print(f"     Clarity score: {underpriced_df['clarity_score'].mean():.2f} (0=I1, 7=IF)")

print("\n   OVERPRICED diamonds (avoid) tend to have:")
print(f"     Cut score: {overpriced_df['cut_score'].mean():.2f}")
print(f"     Color score: {overpriced_df['color_score'].mean():.2f}")
print(f"     Clarity score: {overpriced_df['clarity_score'].mean():.2f}")

print("\n" + "=" * 70)
print("📊 See visualization: /tmp/diamond_mispricing_analysis.png")
print("=" * 70)
