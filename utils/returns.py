import pandas as pd

def compute_returns(prices):
    """
    Computes simple returns from prices.
    """
    return prices.pct_change().dropna()

def compute_portfolio_returns(asset_returns, weights):
    """
    Computes portfolio daily returns based on asset returns and weights.
    """
    return asset_returns.dot(weights)

def calculate_cumulative_returns(returns):
    """
    Computes cumulative returns series (Wealth Index - 1).
    """
    return (1 + returns).cumprod() - 1

def calculate_annualized_metrics(returns, risk_free_rate=0.0):
    """
    Computes annualized return and volatility.
    Returns a dictionary with 'ann_ret' and 'ann_vol'.
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe
    }

def calculate_drawdown_series(returns):
    """
    Computes the drawdown series for a return series.
    """
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns
