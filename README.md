# Advanced Portfolio Analytics

A professional-grade investment analysis dashboard built with Streamlit and Python. This application supports the full portfolio management lifecycle, from construction and optimization to deep performance analysis and tail risk assessment.

## Features

The application is structured into four sequential modules:

### 1. Portfolio Builder
*   **Asset Selection**: Choose from a global universe of assets (Equities, ETFs, Bonds, Crypto).
*   **Weighting Schemes**: Define allocations using Equal Weights or Manual Weights.
*   **Validation**: Ensures fully invested (100%) and valid portfolios before proceeding.

### 2. Analytics
*   **Performance Metrics**: Cumulative returns, Annualized Return/Vol, Sharpe Ratio, Max Drawdown.
*   **Benchmark Comparison**: Relative performance vs S&P 500, Nasdaq, ACWI, etc.
    *   Active Return, Tracking Error, Information Ratio.
*   **Factor Analysis**: Decomposition of returns using:
    *   **CAPM**: Market Beta and Alpha.
    *   **Fama-French 3-Factor**: Market, Size (SMB), and Value (HML) exposure.

### 3. Optimization
*   **Mean-Variance Optimization (MVO)**: Mathematically optimal weight allocation to maximize Sharpe ratio.
*   **Efficient Frontier**: Visualization of the risk-return tradeoff.
*   **Constraints**: Fully invested, long-only, specific asset caps (e.g., max 40%).
*   **Target Beta**: Portfolio beta management relative to the market.

### 4. Risk Analysis
*   **Historical Tail Risk**: VaR (Value at Risk) and CVaR (Conditional VaR) at 95% and 99% confidence levels.
*   **Stress Testing**: Simulation of historical crises:
    *   2007–2009 Subprime Crisis
    *   2020 COVID Shock
    *   2022 Inflation & Rates Shock
*   **Monte Carlo-Simulation**: Future projection of portfolio value with confidence intervals (cone of uncertainty).

## Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Processing**: Pandas, NumPy
*   **Financial Data**: yfinance
*   **Visualization**: Plotly Express & Plotly Graph Objects
*   **Styling**: Custom CSS injection for a premium, institutional dark mode aesthetic.

Navigate through the sidebar stages starting with **Builder** to construct your initial portfolio.

## Structure

*   `app.py`: Main entry point and global configuration.
*   `home.py`: Landing page and workflow overview.
*   `pages/`: Individual application modules (Builder, Analytics, Optimization, Risk).
*   `utils/`: Core logic for data fetching, financial calculations, and optimization algorithms.
