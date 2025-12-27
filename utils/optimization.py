import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st

def get_portfolio_metrics(weights, returns, rf_rate=0.0):
    """
    Computes portfolio return, volatility, and Sharpe ratio.
    Assumes returns are annualized (mean * 252) and cov is annualized.
    """
    # Assuming 'returns' is historical expected return (Series)
    # and we need covariance for risk.
    # Actually, standard MVO uses exp_returns and cov_matrix.
    # We'll adjust arguments to take mean_returns and cov_matrix.
    pass

def run_mean_variance_optimization(current_weights, asset_returns, benchmark_returns=None, max_weight=0.40, beta_target=None, beta_tol=0.2):
    """
    Runs Mean-Variance Optimization to maximize Sharpe Ratio.
    
    Args:
        current_weights (dict): {ticker: weight}
        asset_returns (pd.DataFrame): Daily asset returns.
        benchmark_returns (pd.Series): Daily benchmark returns (required for Beta constraint).
        max_weight (float): Maximum weight per asset (default 0.40).
        beta_target (float): Target Beta (usually current portfolio beta).
        beta_tol (float): Allowed deviation from target beta (+/- 0.2).
        
    Returns:
        dict: Optimized weights {ticker: weight}
        float: Optimized Beta
    """
    tickers = list(current_weights.keys())
    num_assets = len(tickers)
    
    # 1. Inputs (Annualized)
    mean_daily_ret = asset_returns.mean()
    mu = mean_daily_ret * 252
    
    cov_matrix = asset_returns.cov() * 252
    
    # Pre-calculate asset betas if benchmark provided
    asset_betas = []
    if benchmark_returns is not None:
        # Align data
        for t in tickers:
            df = pd.DataFrame({"Asset": asset_returns[t], "Bench": benchmark_returns}).dropna()
            if not df.empty:
                 # Cov/Var
                 beta = df["Asset"].cov(df["Bench"]) / df["Bench"].var()
                 asset_betas.append(beta)
            else:
                 asset_betas.append(1.0) # Fallback
        asset_betas = np.array(asset_betas)
    
    # 2. Objective Function: Maximize Sharpe ratio => Minimize -Sharpe
    # Rf = 0% strictly as per requirement
    def negative_sharpe(w):
        p_ret = np.sum(mu * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return - (p_ret / p_vol) if p_vol > 0 else 0
        
    # 3. Constraints
    constraints = [
        # Fully invested: sum(w) = 1
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    # Beta Constraint (Soft/Hard?)
    # |Beta_opt - Beta_curr| <= 0.2  =>  Beta_curr - 0.2 <= Beta_opt <= Beta_curr + 0.2
    if beta_target is not None and len(asset_betas) == num_assets:
        min_beta = beta_target - beta_tol
        max_beta = beta_target + beta_tol
        
        # Beta_opt = sum(w * asset_betas)
        # Constraint 1: sum(w*beta) >= min_beta
        constraints.append({'type': 'ineq', 'fun': lambda w: np.sum(w * asset_betas) - min_beta})
        # Constraint 2: max_beta >= sum(w*beta) => max_beta - sum(...) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda w: max_beta - np.sum(w * asset_betas)})
        
    # 4. Bounds (Long-only, Max Weight)
    # 0 <= w <= 0.40
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))
    
    # Initial Guess: Equal weights or Current weights
    w0 = np.array([current_weights[t] for t in tickers])
    # Normalize w0 just in case
    w0 = w0 / np.sum(w0)
    
    # 5. Optimization
    result = minimize(negative_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        # Fallback? Or just return warning?
        # Often fails due to strict constraints. Try relaxing or return current?
        # Requirement: "If infeasible, display a clear warning." -> Handled in UI
        return None, None
        
    opt_weights = result.x
    
    # Cleanup small weights
    opt_weights[opt_weights < 0.0001] = 0
    opt_weights = opt_weights / np.sum(opt_weights)
    
    # Output Dictionary
    optimized_weights_dict = {tickers[i]: opt_weights[i] for i in range(num_assets)}
    
    # Calculate Optimized Beta
    opt_beta = np.sum(opt_weights * asset_betas) if len(asset_betas) > 0 else None
    
    return optimized_weights_dict, opt_beta

def get_efficient_frontier(asset_returns, num_points=20):
    """
    Generate efficient frontier points (Volatility, Return) using MVO.
    """
    mean_daily_ret = asset_returns.mean()
    mu = mean_daily_ret * 252
    cov_matrix = asset_returns.cov() * 252
    num_assets = len(asset_returns.columns)
    
    # Find Min Vol Portfolio
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    cons_vol = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1./num_assets,]
    
    min_vol_res = minimize(port_vol, init_guess, method='SLSQP', bounds=bounds, constraints=cons_vol)
    min_vol = min_vol_res.fun
    min_vol_ret = np.sum(mu * min_vol_res.x)
    
    # Find Max Return Portfolio (conceptually 100% in highest return asset, but MVO can imply leverage, here bounds 0-1)
    # Actually simpler: MVO for target return.
    max_ret = mu.max()
    
    target_returns = np.linspace(min_vol_ret, max_ret, num_points)
    frontier_vols = []
    frontier_rets = []
    
    for tr in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * mu) - tr}
        ]
        
        res = minimize(port_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            frontier_vols.append(res.fun)
            frontier_rets.append(tr)
            
    return frontier_vols, frontier_rets
