"""
RiskIntelli: Portfolio Risk Intelligence Platform (Version 1)
-------------------------------------------------------------
Module: portfolio_risk.py

Purpose:
    Calculates key portfolio risk metrics for a user-defined hypothetical
    foreign exchange portfolio using historical daily returns.

Disclaimer:
    This is an educational decision-support project for a college viva / report.
    It does not execute trades, connect to brokers, use real money, provide
    buy/sell advice, or guarantee any investment returns.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def resolve_data_path() -> Path:
    """
    Locates eda_output/daily_returns.csv across standard project directories.
    """
    candidates = [
        Path(__file__).resolve().parent / "eda_output" / "daily_returns.csv",
        Path(__file__).resolve().parent.parent / "eda_output" / "daily_returns.csv",
        Path("eda_output/daily_returns.csv"),
        Path("Project Dataset/eda_output/daily_returns.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate 'eda_output/daily_returns.csv'. "
        "Please ensure EDA.py has been run and generated eda_output."
    )


def validate_inputs(weights: dict, notional: float, available_assets: list) -> None:
    """
    Validates portfolio weights and notional amount:
    1. Notional value must be strictly positive (> 0).
    2. Weights dictionary cannot be empty.
    3. All weights must be non-negative (>= 0).
    4. Weights must sum up to exactly 1.0 (100%) within standard tolerance.
    5. All currency pairs in weights must exist in the historical data columns.
    """
    if notional <= 0:
        raise ValueError(f"Invalid notional value: {notional}. Notional must be greater than zero.")

    if not weights or not isinstance(weights, dict):
        raise ValueError("Weights must be a non-empty dictionary mapping currency pairs to proportions.")

    # Check for negative weights
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Negative weights detected (shorting not allowed in V1): {negative_weights}")

    # Check weight summation (tolerance of 1e-4 for floating-point arithmetic)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-4):
        raise ValueError(
            f"Weights must sum to 1.0 (100%). Current sum: {total_weight:.6f} "
            f"(difference: {abs(total_weight - 1.0):.6f})"
        )

    # Check asset availability
    missing_assets = [pair for pair in weights.keys() if pair not in available_assets]
    if missing_assets:
        raise ValueError(
            f"Assets not found in dataset: {missing_assets}. "
            f"Available assets are: {available_assets}"
        )


def calculate_portfolio_risk(
    weights: dict,
    notional: float = 100000.0,
    returns_df: pd.DataFrame = None,
    data_path: Path = None,
    output_dir: Path = None,
) -> dict:
    """
    Calculates portfolio risk metrics in the required Version 1 order:
      1. Portfolio daily return
      2. Daily volatility (and annualized volatility)
      3. Cumulative return & final portfolio value
      4. Drawdown & Maximum Drawdown
      5. Historical 95% one-day Value at Risk (VaR)
      6. Correlation matrix of portfolio assets

    Returns:
        dict: Complete risk summary structure suitable for API/JSON export.
    """
    # 0. Load data if not supplied
    if returns_df is None:
        if data_path is None:
            data_path = resolve_data_path()
        returns_df = pd.read_csv(data_path, parse_dates=["time"]).set_index("time")

    # Validate inputs
    validate_inputs(weights, notional, list(returns_df.columns))

    # Active asset weights series (filter non-zero weights)
    weight_series = pd.Series({k: v for k, v in weights.items() if v > 0})
    active_assets = list(weight_series.index)
    asset_returns = returns_df[active_assets]

    # -------------------------------------------------------------
    # 1. Portfolio Daily Return:
    #    portfolio_return = sum(weight_i * pair_daily_return_i)
    # -------------------------------------------------------------
    portfolio_daily_returns = asset_returns.dot(weight_series)

    # -------------------------------------------------------------
    # 2. Daily Volatility:
    #    Standard deviation of portfolio daily returns
    #    sigma_daily = std(R_p)
    # -------------------------------------------------------------
    daily_volatility = float(portfolio_daily_returns.std())
    annualized_volatility = float(daily_volatility * np.sqrt(252))

    # -------------------------------------------------------------
    # 3. Cumulative Return:
    #    Cumulative growth of the portfolio over the historical period
    #    CumGrowth_t = prod(1 + R_p)
    # -------------------------------------------------------------
    cumulative_growth_series = (1 + portfolio_daily_returns).cumprod()
    cumulative_return_series = cumulative_growth_series - 1.0
    total_cumulative_return = float(cumulative_return_series.iloc[-1])
    final_portfolio_value = float(notional * cumulative_growth_series.iloc[-1])

    # -------------------------------------------------------------
    # 4. Drawdown:
    #    Decline from the previous cumulative-return peak
    #    Peak_t = max_{s <= t}(CumGrowth_s)
    #    Drawdown_t = (CumGrowth_t - Peak_t) / Peak_t
    # -------------------------------------------------------------
    running_peak = cumulative_growth_series.cummax()
    drawdown_series = (cumulative_growth_series - running_peak) / running_peak
    max_drawdown = float(drawdown_series.min())
    current_drawdown = float(drawdown_series.iloc[-1])
    max_drawdown_amount = float(abs(max_drawdown) * notional)

    # -------------------------------------------------------------
    # 5. Historical 95% One-Day Value at Risk (VaR):
    #    Empirical 95th percentile of historical daily losses
    #    Loss = -R_p
    #    VaR_95 = percentile(losses, 95)
    # -------------------------------------------------------------
    daily_losses = -portfolio_daily_returns
    var_95_daily_pct = float(np.percentile(daily_losses, 95))
    var_95_notional = float(var_95_daily_pct * notional)

    # Secondary reference: 99% VaR and Expected Shortfall (CVaR)
    var_99_daily_pct = float(np.percentile(daily_losses, 99))
    var_99_notional = float(var_99_daily_pct * notional)
    cvar_95_daily_pct = float(daily_losses[daily_losses >= var_95_daily_pct].mean())
    cvar_95_notional = float(cvar_95_daily_pct * notional)

    # -------------------------------------------------------------
    # 6. Correlation Matrix:
    #    Correlation between pairs included in the portfolio
    # -------------------------------------------------------------
    correlation_matrix = asset_returns.corr().to_dict()

    # Determine date span
    start_date = str(portfolio_daily_returns.index.min().date())
    end_date = str(portfolio_daily_returns.index.max().date())
    trading_days = int(len(portfolio_daily_returns))

    # Assemble comprehensive risk summary
    risk_summary = {
        "metadata": {
            "platform": "RiskIntelli - Portfolio Risk Intelligence",
            "version": "1.0",
            "scope": "Educational Decision-Support",
            "start_date": start_date,
            "end_date": end_date,
            "total_trading_days": trading_days,
            "notional_value": notional,
            "portfolio_weights": weights,
        },
        "metrics": {
            "daily_volatility": {
                "percentage": round(daily_volatility * 100, 4),
                "decimal": round(daily_volatility, 6),
                "description": "Standard deviation of portfolio daily returns",
            },
            "annualized_volatility": {
                "percentage": round(annualized_volatility * 100, 4),
                "decimal": round(annualized_volatility, 6),
                "description": "Annualized standard deviation (daily_vol * sqrt(252))",
            },
            "cumulative_return": {
                "total_percentage": round(total_cumulative_return * 100, 4),
                "final_portfolio_value": round(final_portfolio_value, 2),
                "description": "Historical cumulative portfolio growth over the observation period",
            },
            "drawdown": {
                "max_drawdown_percentage": round(max_drawdown * 100, 4),
                "max_drawdown_amount": round(max_drawdown_amount, 2),
                "current_drawdown_percentage": round(current_drawdown * 100, 4),
                "description": "Decline from the previous cumulative return peak",
            },
            "historical_var_95": {
                "one_day_percentage": round(var_95_daily_pct * 100, 4),
                "one_day_amount": round(var_95_notional, 2),
                "disclaimer": (
                    "Historical 95% one-day estimate: On 95% of trading days, daily loss "
                    "did not exceed this threshold. This is an empirical risk estimate, "
                    "NOT the maximum possible loss."
                ),
            },
            "additional_risk_benchmarks": {
                "historical_var_99_percentage": round(var_99_daily_pct * 100, 4),
                "historical_var_99_amount": round(var_99_notional, 2),
                "expected_shortfall_cvar_95_percentage": round(cvar_95_daily_pct * 100, 4),
                "expected_shortfall_cvar_95_amount": round(cvar_95_notional, 2),
            },
        },
        "correlation_matrix": correlation_matrix,
    }

    # Save summary if output directory is defined or standard location
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "risk_output"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "risk_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(risk_summary, f, indent=4)

    # Also save portfolio daily performance series for charts/dashboard
    performance_df = pd.DataFrame({
        "time": portfolio_daily_returns.index,
        "daily_return": portfolio_daily_returns.values,
        "cumulative_return": cumulative_return_series.values,
        "drawdown": drawdown_series.values,
        "portfolio_value": (notional * cumulative_growth_series).values,
    })
    performance_df.to_csv(output_dir / "portfolio_performance.csv", index=False)

    return risk_summary


def print_risk_report(summary: dict) -> None:
    """Prints a clear, formatted terminal risk report for viva presentation."""
    meta = summary["metadata"]
    m = summary["metrics"]

    print("=" * 72)
    print("   RiskIntelli: Portfolio Risk Intelligence Platform (Version 1)")
    print("=" * 72)
    print(f"Observation Period : {meta['start_date']} to {meta['end_date']} ({meta['total_trading_days']} trading days)")
    print(f"Portfolio Notional : ${meta['notional_value']:,.2f}")
    print("\nPortfolio Asset Allocation:")
    for pair, weight in meta["portfolio_weights"].items():
        print(f"  - {pair:<8}: {weight * 100:6.2f}%")

    print("\n" + "-" * 72)
    print("Core Risk Metrics (Calculated in V1 Order):")
    print("-" * 72)

    print(f"1. Daily Volatility      : {m['daily_volatility']['percentage']:.4f}% daily "
          f"(Annualized: {m['annualized_volatility']['percentage']:.2f}%)")

    print(f"2. Cumulative Return     : {m['cumulative_return']['total_percentage']:+.2f}% "
          f"-> Final Portfolio Value: ${m['cumulative_return']['final_portfolio_value']:,.2f}")

    print(f"3. Maximum Drawdown      : {m['drawdown']['max_drawdown_percentage']:.2f}% "
          f"(Max Drop: ${m['drawdown']['max_drawdown_amount']:,.2f})")
    print(f"   Current Drawdown      : {m['drawdown']['current_drawdown_percentage']:.2f}%")

    print(f"4. Historical 95% 1-Day VaR: {m['historical_var_95']['one_day_percentage']:.4f}% "
          f"(VaR Amount: ${m['historical_var_95']['one_day_amount']:,.2f})")
    print(f"   [Label/Disclaimer]    : {m['historical_var_95']['disclaimer']}")

    print("\nAdditional Benchmark Metrics:")
    print(f"   Historical 99% 1-Day VaR: {m['additional_risk_benchmarks']['historical_var_99_percentage']:.4f}% "
          f"(${m['additional_risk_benchmarks']['historical_var_99_amount']:,.2f})")
    print(f"   95% Expected Shortfall  : {m['additional_risk_benchmarks']['expected_shortfall_cvar_95_percentage']:.4f}% "
          f"(${m['additional_risk_benchmarks']['expected_shortfall_cvar_95_amount']:,.2f})")
    print("=" * 72)


if __name__ == "__main__":
    # Standard example portfolio from specification
    example_weights = {
        "EURUSD": 0.20,
        "GBPUSD": 0.15,
        "USDJPY": 0.15,
        "USDCHF": 0.10,
        "AUDUSD": 0.15,
        "USDCAD": 0.15,
        "NZDUSD": 0.10,
    }
    example_notional = 100000.0

    print("Running RiskIntelli Portfolio Risk Engine with example portfolio...\n")
    risk_results = calculate_portfolio_risk(
        weights=example_weights,
        notional=example_notional,
    )
    print_risk_report(risk_results)
    print("\nResults successfully saved to 'risk_output/risk_summary.json' and 'risk_output/portfolio_performance.csv'.")

