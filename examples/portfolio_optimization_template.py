#!/usr/bin/env python
"""
Portfolio Optimization Template using Deferred Expressions

This template demonstrates how to:
1. Filter items by multiple quality constraints
2. Score and rank items by efficiency metrics
3. Ensure diversity across categories
4. Optimize selection within budget constraints
5. Cache results for performance

Pattern: Multi-criteria optimization with diversity requirements
Use case: Asset selection, resource allocation, inventory optimization
"""

import xorq.api as xo
from xorq.vendor import ibis
from xorq.caching import ParquetCache
from xorq.common.utils.ibis_utils import from_ibis

# ============================================================================
# CONFIGURATION - Customize these for your use case
# ============================================================================

BUDGET = 10000  # Maximum total cost/investment
MIN_QUALITY_THRESHOLD = 3  # Minimum acceptable quality score
REQUIRED_CATEGORIES = ["Category_A", "Category_B", "Category_C"]  # Diversity requirement

# Quality grading mappings
QUALITY_GRADES = {
    "Premium": 5,
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Basic": 1
}

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================

def load_and_filter_data():
    """Load source data and apply quality constraints."""

    # Connect to backend
    con = xo.connect()

    # Load from catalog (replace with your source)
    # source = xo.catalog.get("your-data-source")
    # OR load from table
    source = con.table("items")

    # Check schema first (MANDATORY)
    print("Schema:")
    print(source.schema())

    # Apply quality constraints
    filtered = source.filter([
        source.quality_score >= MIN_QUALITY_THRESHOLD,
        source.price <= BUDGET,  # Individual items within budget
        source.price > 0,  # Valid prices only
        source.quantity_available > 0  # In stock
    ])

    return filtered


# ============================================================================
# STEP 2: Add Scoring and Ranking
# ============================================================================

def add_scoring_metrics(expr):
    """Add efficiency metrics and quality scores."""

    # Calculate efficiency (value per cost)
    enriched = expr.mutate(
        efficiency=expr.value / expr.price,

        # Map quality grades to numeric scores
        quality_rank=ibis.cases(
            *[(expr.quality == grade, score)
              for grade, score in QUALITY_GRADES.items()],
            else_=0
        ),

        # Calculate composite score
        # Adjust weights based on your priorities
        composite_score=(
            (expr.value / expr.price) * 0.4 +  # Efficiency weight
            ibis.cases(
                *[(expr.quality == grade, score)
                  for grade, score in QUALITY_GRADES.items()],
                else_=0
            ) * 0.3 +  # Quality weight
            expr.user_rating * 0.3  # Rating weight
        )
    )

    return enriched


# ============================================================================
# STEP 3: Multi-Criteria Selection with Diversity
# ============================================================================

def select_diverse_portfolio(expr, categories=REQUIRED_CATEGORIES):
    """
    Select best items ensuring at least one from each category.
    This implements a diversity constraint for the portfolio.
    """

    selections = []

    # Phase 1: Mandatory selections (one from each category)
    for category in categories:
        best_in_category = (
            expr
            .filter(expr.category == category)
            .order_by(expr.efficiency.desc())
            .limit(1)
            .mutate(
                selection_phase=ibis.literal("mandatory"),
                selection_reason=ibis.literal(f"Best in {category}")
            )
        )
        selections.append(best_in_category)

    # Union mandatory selections
    mandatory_portfolio = ibis.union(*selections) if selections else None

    # Phase 2: Additional selections for remaining budget
    # Note: In practice, you'd implement knapsack algorithm via UDF
    # This is a simplified greedy approach

    if mandatory_portfolio:
        # Exclude already selected items
        remaining = expr.anti_join(
            mandatory_portfolio.select("item_id"),
            predicates=[expr.item_id == mandatory_portfolio.item_id]
        )

        # Select additional high-efficiency items
        additional = (
            remaining
            .order_by(remaining.efficiency.desc())
            .limit(100)  # Take top N candidates
            .mutate(
                selection_phase=ibis.literal("optimization"),
                selection_reason=ibis.literal("High efficiency")
            )
        )

        # Combine portfolios
        final_portfolio = ibis.union(mandatory_portfolio, additional)
    else:
        # No mandatory categories, pure optimization
        final_portfolio = (
            expr
            .order_by(expr.efficiency.desc())
            .limit(100)
            .mutate(
                selection_phase=ibis.literal("optimization"),
                selection_reason=ibis.literal("Top efficiency")
            )
        )

    return final_portfolio


# ============================================================================
# STEP 4: Apply Budget Constraint
# ============================================================================

def apply_budget_constraint(portfolio, budget=BUDGET):
    """
    Apply budget constraint using cumulative cost window function.
    Note: This is approximate - for exact knapsack, use a UDF.
    """

    # Add cumulative cost
    with_cumsum = portfolio.mutate(
        cumulative_cost=portfolio.price.sum().over(
            ibis.window(
                order_by=[portfolio.selection_phase, portfolio.efficiency.desc()],
                preceding=0,
                following=0
            )
        )
    )

    # Filter to stay within budget
    # This is a simplified approach - production code would use dynamic programming
    budget_constrained = with_cumsum.filter(
        with_cumsum.cumulative_cost <= budget
    )

    return budget_constrained


# ============================================================================
# STEP 5: Add Analytics and Cache
# ============================================================================

def add_portfolio_analytics(portfolio):
    """Add analytics metrics to the portfolio."""

    # Add portfolio-level metrics
    with_analytics = portfolio.mutate(
        # Percentage of budget used by each item
        budget_percentage=(portfolio.price / BUDGET * 100).round(2),

        # Rank by value contribution
        value_rank=ibis.row_number().over(
            order_by=[portfolio.value.desc()]
        ),

        # Efficiency percentile
        efficiency_percentile=ibis.percent_rank().over(
            order_by=[portfolio.efficiency]
        ).round(3)
    )

    return with_analytics


def create_portfolio_summary(portfolio):
    """Create summary statistics for the portfolio."""

    summary = portfolio.aggregate([
        portfolio.price.sum().name("total_cost"),
        portfolio.value.sum().name("total_value"),
        portfolio.count().name("item_count"),
        portfolio.category.nunique().name("unique_categories"),
        portfolio.efficiency.mean().round(4).name("avg_efficiency"),
        portfolio.quality_rank.mean().round(2).name("avg_quality"),
        (portfolio.value.sum() / portfolio.price.sum()).round(4).name("portfolio_roi")
    ])

    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_optimization_pipeline():
    """Build the complete portfolio optimization pipeline."""

    # Step 1: Load and filter
    filtered_items = load_and_filter_data()

    # Step 2: Add scoring
    scored_items = add_scoring_metrics(filtered_items)

    # Step 3: Select diverse portfolio
    portfolio = select_diverse_portfolio(scored_items)

    # Step 4: Apply budget constraint
    budget_portfolio = apply_budget_constraint(portfolio)

    # Step 5: Add analytics
    final_portfolio = add_portfolio_analytics(budget_portfolio)

    # Step 6: Cache for performance
    cached_portfolio = from_ibis(final_portfolio).cache(
        ParquetCache.from_kwargs()
    )

    # Create summary
    summary = create_portfolio_summary(final_portfolio)
    cached_summary = from_ibis(summary).cache(
        ParquetCache.from_kwargs()
    )

    return {
        'portfolio': cached_portfolio,
        'summary': cached_summary
    }


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # For direct execution
    print("Building portfolio optimization pipeline...")
    pipeline = build_optimization_pipeline()

    print("\nExecuting portfolio selection...")
    portfolio_df = pipeline['portfolio'].execute()
    summary_df = pipeline['summary'].execute()

    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*60)

    print("\nSummary Statistics:")
    print(summary_df.to_string())

    print(f"\nTop 10 Selected Items:")
    print(portfolio_df.head(10)[['item_id', 'category', 'price',
                                  'value', 'efficiency', 'selection_reason']])

    print("\nCategory Coverage:")
    category_counts = portfolio_df.groupby('category').size()
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} items")

else:
    # For xorq build
    pipeline = build_optimization_pipeline()
    expr = pipeline['portfolio']  # Main expression to build
    summary_expr = pipeline['summary']  # Secondary expression